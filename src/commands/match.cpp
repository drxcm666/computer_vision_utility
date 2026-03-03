#include "cvtool/commands/match.hpp"
#include "cvtool/core/template_match.hpp"
#include "cvtool/core/image_io.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/rois_edges.hpp"
#include "cvtool/core/match/match_json.hpp"
#include "cvtool/core/match/match_heatmap.hpp"
#include "cvtool/core/match/match_render.hpp"
#include "cvtool/core/image_convert.hpp"
#include "cvtool/core/match/match_search_ms.hpp"
#include "cvtool/core/match/match_prepare.hpp"
#include "cvtool/commands/match_validate.hpp"

#include <opencv2/imgproc.hpp>

#include <fmt/core.h>

#include <string_view>
#include <charconv>

#include <cmath>
#include <array>

cvtool::core::ExitCode run_match(const cvtool::cmd::MatchOptions &opt)
{
    cv::Mat scene, templ;
    std::string err;

    auto read_code = cvtool::core::image_io::read_image(opt.in_path, scene, err);
    if (read_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return read_code;
    }
    read_code = cvtool::core::image_io::read_image(opt.templ_path, templ, err);
    if (read_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return read_code;
    }

    int method{0};
    double scale_min{1.0}, scale_max{1.0}, scale_step{1.0};
    int count{1};
    int per_scale_top{0};

    auto v = cvtool::cmd::match_validate::validate_match_options(
        opt, method, scale_min, scale_max, scale_step, count, per_scale_top, err);
    if (v != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return v;
    }

    cv::Mat scene_proc, templ_proc;
    auto c1 = cvtool::core::match_preparate::preparate_for_match(scene, opt.mode, scene_proc, err);
    if (c1 != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return c1;
    }

    auto c2 = cvtool::core::match_preparate::preparate_for_match(templ, opt.mode, templ_proc, err);
    if (c2 != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return c2;
    }

    if (templ_proc.cols > scene_proc.cols || templ_proc.rows > scene_proc.rows)
    {
        fmt::println(stderr, "error: template larger than scene (templ: {}x{}, scene: {}x{})",
                     templ_proc.cols, templ_proc.rows, scene_proc.cols, scene_proc.rows);

        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    fmt::println("command: match");
    fmt::println("in: {}", opt.in_path);
    fmt::println("templ: {}", opt.templ_path);
    fmt::println("out: {}", opt.out_path);
    fmt::println("mode: {}", opt.mode);
    fmt::println("method: {}", opt.method);
    fmt::println("templ_size: {}x{}", templ_proc.cols, templ_proc.rows);
    fmt::println("scene_size: {}x{}", scene_proc.cols, scene_proc.rows);
    fmt::println("params: max_results={} min_score={:.2f} nms={:.2f} draw={} thickness={} font_scale={:.2f} roi={} json={} heatmap={}",
                 opt.max_results,
                 opt.min_score,
                 opt.nms,
                 opt.draw,
                 opt.thickness,
                 opt.font_scale,
                 opt.roi.empty() ? "none" : opt.roi,
                 opt.json_path.empty() ? "none" : opt.json_path,
                 opt.heatmap_path.empty() ? "none" : opt.heatmap_path);
    fmt::println("scales: min={:.2f}, max={:.2f}, step={:.2f}, count={}", scale_min, scale_max, scale_step, count);
    fmt::println("per_scale_top: {}", per_scale_top);

    cv::Rect roi;
    if (!opt.roi.empty())
    {
        auto roi_code = cvtool::core::validate::validate_roi(opt.roi, roi, err);
        if (roi_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return roi_code;
        }

        cv::Rect bounds{0, 0, scene_proc.cols, scene_proc.rows};
        if ((roi & bounds) != roi)
        {
            fmt::println(stderr, "error: roi out of bounds");
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
    }

    std::vector<cv::Rect> rois;
    bool roi_fallback_used{false};
    cv::Mat scene_gray = cvtool::core::img::to_gray(scene_proc);

    if (opt.roi_auto == "contours")
    {
        fmt::println(stderr, "error: roi-auto=contours not implemented yet");
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    if (opt.roi.empty() && (opt.roi_auto == "edges"))
    {
        cvtool::core::roi_edges::RoiEdgesParams p;
        p.low = opt.roi_edges_low;
        p.high = opt.roi_edges_high;
        p.blur_k = opt.roi_edges_blur_k;
        p.roi_max = opt.roi_max;
        p.min_area = opt.roi_min_area;
        p.pad = opt.roi_pad;
        p.merge_iou = opt.roi_merge_iou;

        rois = cvtool::core::roi_edges::build_rois_edges(scene_gray, p);

        if (rois.empty())
        {
            if (opt.roi_fallback)
            {
                rois.push_back(cv::Rect{0, 0, scene_gray.cols, scene_gray.rows});
                roi_fallback_used = true;
            }
            else
            {
                fmt::println(stderr, "error: no ROI found");
                return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
            }
        }

        fmt::println("roi_auto: edges");
        fmt::println("roi_count: {}", (int)rois.size());
        for (int i = 0; i < (int)rois.size(); ++i)
            fmt::println("roi[{}]: x={} y={} w={} h={}", i, rois[i].x, rois[i].y, rois[i].width, rois[i].height);

        if (roi_fallback_used)
            fmt::println("roi_fallback_used: true");
    }

    std::vector<cv::Rect> search_rois;
    std::string roi_source;
    if (!opt.roi.empty())
    {
        search_rois.push_back(roi);
    }
    else if (opt.roi_auto == "edges")
    {
        search_rois = rois;
    }
    else
    {
        search_rois.push_back({0, 0, scene_proc.cols, scene_proc.rows});
    }

    int w_min = static_cast<int>(std::lround(templ_proc.cols * scale_min));
    int h_min = static_cast<int>(std::lround(templ_proc.rows * scale_min));
    search_rois.erase(std::remove_if(search_rois.begin(), search_rois.end(),
                                     [&](const cv::Rect &r)
                                     { return r.width < w_min || r.height < h_min; }),
                      search_rois.end());
    if (search_rois.empty())
    {
        if (opt.roi_auto == "edges" && opt.roi_fallback)
        {
            search_rois.push_back({0, 0, scene_proc.cols, scene_proc.rows});
            roi_fallback_used = true;
            fmt::println("roi_fallback_used: true");
        }
        else
        {
            fmt::println(stderr, "error: no ROI found");
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
    }

    if (!opt.roi.empty())
        roi_source = "manual";
    else if (roi_fallback_used)
        roi_source = "fallback";
    else if (opt.roi_auto == "edges")
        roi_source = "edges";
    else
        roi_source = "full";

    std::vector<cvtool::core::templ_match::MatchBest> all;
    all.reserve((std::size_t)count * (std::size_t)per_scale_top * (std::size_t)search_rois.size());
    int valid_scales{0};
    cv::Mat best_result;
    const bool need_heatmap = !opt.heatmap_path.empty();

    auto multiscale_code = cvtool::core::match_search_ms::search_multiscale(
        scene_proc, templ_proc, method, search_rois,
        scale_min, scale_step, count, per_scale_top,
        opt.min_score, need_heatmap, all, best_result, valid_scales,
        err);

    if (multiscale_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return multiscale_code;
    }

    auto hits_topk = cvtool::core::templ_match::nms_iou(all, opt.nms, opt.max_results);

    if (!opt.heatmap_path.empty())
    {
        auto heat_code = cvtool::core::match_heatmap::write_heatmap(best_result, method, opt.heatmap_path, err);
        if (heat_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return heat_code;
        }
    }
    if (!opt.json_path.empty())
    {
        const auto json_code = cvtool::core::match_json::write_match_json(
            opt,
            scene_proc.size(),
            templ_proc.size(),
            hits_topk,
            search_rois,
            roi_fallback_used,
            roi_source,
            err);
        if (json_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return json_code;
        }
    }

    cv::Mat vis;

    auto render_code = cvtool::core::match_render::render(
        search_rois, hits_topk, scene, opt.draw_roi, opt.draw, opt.thickness, opt.font_scale, vis, err);

    if (render_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return render_code;
    }

    if (hits_topk.empty())
    {
        fmt::println("status: ok\nfound: 0");
        const auto json_code = cvtool::core::image_io::write_image(opt.out_path, vis, err);
        if (json_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return json_code;
        }
        return cvtool::core::ExitCode::Ok;
    }

    fmt::println("status: ok\nfound: {}", hits_topk.size());
    if (!hits_topk.empty())
    {
        fmt::println("best: conf={:.2f} raw={:.4f} at x={} y={} scale={:.2f}",
                     hits_topk[0].confidence, hits_topk[0].raw_score,
                     hits_topk[0].bbox.x, hits_topk[0].bbox.y,
                     hits_topk[0].scale);
    }

    const auto write_code = cvtool::core::image_io::write_image(opt.out_path, vis, err);
    if (write_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return write_code;
    }

    return cvtool::core::ExitCode::Ok;
}