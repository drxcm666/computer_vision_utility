#include "cvtool/commands/match.hpp"
#include "cvtool/core/template_match.hpp"
#include "cvtool/core/image_io.hpp"
#include "cvtool/core/validate.hpp"

#include <opencv2/imgproc.hpp>

#include <fmt/core.h>

#include <nlohmann/json.hpp>

#include <string_view>
#include <charconv>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <array>

static cv::Mat to_gray(const cv::Mat &img)
{
    cv::Mat gray;
    if (img.channels() == 1)
    {
        return img;
    }
    else if (img.channels() == 3)
    {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else if (img.channels() == 4)
    {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        return cv::Mat{};
    }

    return gray;
}

static bool to_bgr(const cv::Mat &img, cv::Mat &dst, std::string &err)
{
    switch (img.channels())
    {
    case 1:
        cv::cvtColor(img, dst, cv::COLOR_GRAY2BGR);
        return true;
    case 3:
        dst = img.clone();
        return true;
    case 4:
        cv::cvtColor(img, dst, cv::COLOR_BGRA2BGR);
        return true;
    default:
        err = fmt::format("error: unsupported channels: {}", img.channels());
        return false;
    }
}

static bool prepare_for_match(const cv::Mat &img, cv::Mat &out, std::string_view mode, std::string &err)
{
    if (mode == "gray")
    {
        out = to_gray(img);
        if (!out.empty())
            return true;
        err = fmt::format("can't convert to gray (channels={})", img.channels());
        return false;
    }

    return to_bgr(img, out, err);
}

static cvtool::core::ExitCode make_heatmap(
    const cv::Mat &result,
    int method,
    const std::string &out_path,
    std::string &err)
{
    if (result.empty())
    {
        err = fmt::format("error: heatmap requested but result matrix is empty");
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    cv::Mat heat = result.clone();
    if (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED)
    {
        heat = 1.0 - heat;
    }

    cv::Mat heat_u8;
    cv::normalize(heat, heat_u8, 0, 255, cv::NORM_MINMAX);
    heat_u8.convertTo(heat_u8, CV_8U);

    cv::Mat heat_color;
    cv::applyColorMap(heat_u8, heat_color, cv::COLORMAP_JET);

    const auto write_code = cvtool::core::image_io::write_image(out_path, heat_color, err);
    if (write_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return write_code;
    }

    return cvtool::core::ExitCode::Ok;
}

static cvtool::core::ExitCode write_match_json(
    const cvtool::cmd::MatchOptions &opt,
    const cv::Size &scene_size,
    const cv::Size &templ_size,
    const std::vector<cvtool::core::templ_match::MatchBest> &hits,
    const std::vector<cv::Rect> &rois,
    bool roi_fallback_used,
    const std::string &roi_source,
    std::string &err)
{
    nlohmann::ordered_json j;

    j["command"] = "match";
    j["input"] = opt.in_path;
    j["template"] = opt.templ_path;
    j["output"] = opt.out_path;
    j["params"] = {{"mode", opt.mode}, {"method", opt.method}, {"max_results", opt.max_results}, {"min_score", opt.min_score}, {"nms", opt.nms}, {"draw", opt.draw}, {"thickness", opt.thickness}, {"font_scale", opt.font_scale}};
    j["template_size"] = {{"w", templ_size.width}, {"h", templ_size.height}};
    j["scene_size"] = {{"w", scene_size.width}, {"h", scene_size.height}};

    if (!opt.roi.empty())
    {
        cv::Rect r;
        std::string err_dummy;
        if (cvtool::core::validate::validate_roi(opt.roi, r, err_dummy) == cvtool::core::ExitCode::Ok)
        {
            j["params"]["roi"] = {{"x", r.x}, {"y", r.y}, {"w", r.width}, {"h", r.height}};
        }
        else
        {
            j["params"]["roi"] = nlohmann::ordered_json::object();
        }
    }
    else
    {
        j["params"]["roi"] = nlohmann::ordered_json::object();
    }

    j["roi_auto"] = opt.roi_auto;
    j["roi_fallback"] = roi_fallback_used;
    j["rois"] = nlohmann::json::array();
    for (auto &r : rois)
    {
        j["rois"].push_back({
            {"x", r.x}, {"y", r.y}, {"w", r.width}, {"h", r.height}, {"source", roi_source}
        });
    }

    j["matches"] = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(hits.size()); ++i)
    {
        const auto &h = hits[i];
        j["matches"].push_back({{"id", i},
                                {"bbox", {{"x", h.bbox.x}, {"y", h.bbox.y}, {"w", h.bbox.width}, {"h", h.bbox.height}}},
                                {"raw_score", h.raw_score},
                                {"confidence", h.confidence},
                                {"scale", h.scale},
                                {"template_size", {{"w", h.template_size.width}, {"h", h.template_size.height}}}});
    }
    j["stats"] = {{"found", (int)hits.size()}};

    std::ofstream file(opt.json_path);
    if (!file)
    {
        fmt::println(stderr, "error: cannot open json output '{}'", opt.json_path);
        return cvtool::core::ExitCode::CannotWriteOutput;
    }

    file << std::setw(4) << j << '\n';
    if (!file.good())
    {
        fmt::println(stderr, "error: failed to write json output: {}", opt.json_path);
        return cvtool::core::ExitCode::CannotWriteOutput;
    }

    return cvtool::core::ExitCode::Ok;
}

static double iou_rect(const cv::Rect &a, const cv::Rect &b)
{
    cv::Rect intersection = a & b;
    const double interA = static_cast<double>(intersection.area());
    double unionA = static_cast<double>(a.area()) + static_cast<double>(b.area()) - interA;
    double proc{0.0};

    if (unionA <= 0.0) return 0.0;

    return interA / unionA;
}

static cv::Rect pad_clamp(cv::Rect r, int pad, const cv::Rect &bounds)
{
    r.x -= pad;
    r.y -= pad;
    r.width += 2 * pad;
    r.height += 2 * pad;

    cv::Rect result = r & bounds;

    return result;
}

static std::vector<cv::Rect> merge_roi_iou(std::vector<cv::Rect> rois, double merge_iou)
{
    for (int iter = 0; iter < 4; iter++)
    {
        std::vector<cv::Rect> result;
        bool changed = false;

        for (int i = 0; i < rois.size(); i++)
        {
            if (result.empty())
            {
                result.emplace_back(rois[i]);
                continue;
            }

            bool merged{false};
            for (int j = 0; j < result.size(); j++)
            {
                if (iou_rect(result[j], rois[i]) > merge_iou)
                {
                    result[j] |= rois[i];
                    merged = true;
                    changed = true;

                    break;
                }
            }
            if (!merged)
                result.emplace_back(rois[i]);
        }
        rois = std::move(result);
        if (!changed)
            break;
    }

    return rois;
}

static std::vector<cv::Rect> build_rois_edges(
    const cv::Mat &scene_gray,
    int low, int high, int blur_k,
    int roi_max, double min_area,
    int roi_pad, double roi_merge_iou)
{
    std::vector<cv::Rect> rois;
    if (scene_gray.empty()) return rois;
    
    cv::Mat blur;
    if (blur_k > 0){
        cv::GaussianBlur(scene_gray, blur, cv::Size(blur_k, blur_k), 0, 0);
    }
    else blur = scene_gray;

    cv::Mat edges;
    cv::Canny(blur, edges, low, high);
    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    rois.reserve(contours.size());

    cv::Size s = scene_gray.size();
    double scene_area = s.width * s.height;
    double min_area_px = min_area * scene_area;
    for (auto &c : contours)
    {
        cv::Rect bounding_rect = cv::boundingRect(c);

        if (static_cast<double>(bounding_rect.area()) <= min_area_px) continue;

        rois.push_back(bounding_rect);
    }

    std::sort(rois.begin(), rois.end(), [](const cv::Rect &a, const cv::Rect &b)
              { return a.area() > b.area(); });

    if (rois.size() > roi_max * 4)
    {
        rois.resize(roi_max * 4);
    }

    std::vector<cv::Rect> merged = merge_roi_iou(rois, roi_merge_iou);
    std::sort(merged.begin(), merged.end(), [](const cv::Rect &a, const cv::Rect &b)
              { return a.area() > b.area(); });
    if (merged.size() > roi_max)
    {
        merged.resize(roi_max);
    }

    cv::Rect bounds{0, 0, scene_gray.cols, scene_gray.rows};
    for (auto &r : merged)
    {
        r = pad_clamp(r, roi_pad, bounds);
    }

    return merged;
}

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

    auto mins_code = cvtool::core::validate::validate_01("min-score", opt.min_score, err);
    if (mins_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return mins_code;
    }
    auto nms_code = cvtool::core::validate::validate_01("nms", opt.nms, err);
    if (nms_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return nms_code;
    }
    auto maxres_code = cvtool::core::validate::validate_max_results(opt.max_results, err);
    if (maxres_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return maxres_code;
    }
    auto mode_code = cvtool::core::validate::validate_mode_match(opt.mode, err);
    if (mode_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return mode_code;
    }
    int method{0};
    auto method_code = cvtool::core::validate::validate_method_match(opt.method, method, err);
    if (method_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return method_code;
    }
    auto draw_code = cvtool::core::validate::validate_draw_match(opt.draw, err);
    if (draw_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return draw_code;
    }
    auto thickness_code = cvtool::core::validate::validate_thickness(opt.thickness, err);
    if (thickness_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return thickness_code;
    }
    auto fs_code = cvtool::core::validate::validate_font_scale(opt.font_scale, err);
    if (fs_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return fs_code;
    }
    auto roi_merge_code = cvtool::core::validate::validate_01("roi-merge-iou", opt.roi_merge_iou, err);
    if (roi_merge_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return roi_merge_code;
    }
    auto minarea_code = cvtool::core::validate::validate_01("roi-min-area", opt.roi_min_area, err);
    if (minarea_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return minarea_code;
    }
    auto max_code = cvtool::core::validate::validate_max_results(opt.roi_max, err);
    if (max_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return max_code;
    }
    auto pad_code = cvtool::core::validate::validate_nonneg("roi-pad", opt.roi_pad, err);
    if (pad_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return pad_code;
    }
    auto blur_code = cvtool::core::validate::validate_blur_k(opt.roi_edges_blur_k, err);
    if (blur_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return blur_code;
    }
    auto thr_code = cvtool::core::validate::validate_thresholds(opt.roi_edges_low, opt.roi_edges_high, err);
    if (thr_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return thr_code;
    }

    double scale_min, scale_max, scale_step;
    auto scale_code = cvtool::core::validate::validate_scale_range(opt.scales, scale_min, scale_max, scale_step, err);
    if (scale_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return scale_code;
    }
    const int count = 1 + (static_cast<int>(std::floor((scale_max - scale_min) / scale_step + 1e-9)));
    if (opt.max_scales > 0 && count > opt.max_scales)
    {
        fmt::println(stderr, "error: too many scales (count={}, max={})", count, opt.max_scales);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    const int per_scale_top = (opt.per_scale_top > 0) ? opt.per_scale_top : opt.max_results * 5;

    cv::Mat scene_proc, templ_proc;
    if (!prepare_for_match(scene, scene_proc, opt.mode, err) || !prepare_for_match(templ, templ_proc, opt.mode, err))
    {
        fmt::println(stderr, "error: can't prepare images for mode: {}", opt.mode);
        if (!err.empty())
            fmt::println(stderr, "{}", err);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
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
    cv::Mat scene_gray = to_gray(scene_proc);

    if (opt.roi_auto == "contours")
    {
        fmt::println(stderr, "error: roi-auto=contours not implemented yet");
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    if (opt.roi.empty() && (opt.roi_auto == "edges"))
    {
        rois = build_rois_edges(
            scene_gray, opt.roi_edges_low,
            opt.roi_edges_high, opt.roi_edges_blur_k,
            opt.roi_max, opt.roi_min_area,
            opt.roi_pad, opt.roi_merge_iou);

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

        if (roi_fallback_used) fmt::println("roi_fallback_used: true");
    }

    std::vector<cv::Rect> search_rois;
    std::string roi_source;
    if (!opt.roi.empty()) {
        search_rois.push_back(roi);
    }
    else if (opt.roi_auto == "edges") {
        search_rois = rois;
    }
    else {
        search_rois.push_back({0, 0, scene_proc.cols, scene_proc.rows});
    }

    int w_min = static_cast<int>(std::lround(templ_proc.cols * scale_min));
    int h_min = static_cast<int>(std::lround(templ_proc.rows * scale_min));
    search_rois.erase(std::remove_if(search_rois.begin(), search_rois.end(), 
        [&](const cv::Rect &r){ 
            return r.width < w_min || r.height < h_min; }), 
            search_rois.end()
    );
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
    double best_conf{-1.0};

    cv::Mat result_s;
    cv::Mat result_buffer;
    for (auto &r : search_rois)
    {
        const cv::Mat sub_scene = scene_proc(r);

        for (int i = 0; i < count; i++)
        {
            const double scale = scale_min + i * scale_step;
            const int w_s = static_cast<int>(std::lround(templ_proc.cols * scale));
            const int h_s = static_cast<int>(std::lround(templ_proc.rows * scale));

            if (w_s < 2 || h_s < 2)
                continue;
            if (w_s > sub_scene.cols || h_s > sub_scene.rows)
                continue;

            valid_scales++;

            cv::Mat templ_s;
            if (std::abs(scale - 1) < 1e-5)
            {
                templ_s = templ_proc;
            }
            else
            {
                const auto interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;
                cv::resize(templ_proc, templ_s, cv::Size(w_s, h_s), 0, 0, interp);
            }
            
            cv::Mat *out_res_s = opt.heatmap_path.empty() ? nullptr : &result_buffer;
            auto cands = cvtool::core::templ_match::match_topk(sub_scene, templ_s, method, per_scale_top, opt.min_score, out_res_s);

            for (auto &hh : cands)
            {
                hh.scale = scale;
                hh.template_size = templ_s.size();
                hh.bbox.x += r.x;
                hh.bbox.y += r.y;
            }

            all.insert(all.end(), cands.begin(), cands.end());

            if (!opt.heatmap_path.empty())
            {
                bool is_new_record = false;

                if (best_result.empty() && !result_buffer.empty())
                    is_new_record = true;

                if (!cands.empty() && cands[0].confidence > best_conf)
                    is_new_record = true;

                if (is_new_record)
                {
                    if (!cands.empty()) best_conf = cands[0].confidence;

                    best_result = result_buffer.clone();
                }
            }
        }
    }

    if (valid_scales == 0)
    {
        fmt::println(stderr, "error: no valid scales (template never fits into scene/roi)");
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    auto hits_topk = cvtool::core::templ_match::nms_iou(all, opt.nms, opt.max_results);

    if (!opt.heatmap_path.empty())
    {
        auto heat_code = make_heatmap(best_result, method, opt.heatmap_path, err);
        if (heat_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return heat_code;
        }
    }
    if (!opt.json_path.empty())
    {
        const auto json_code = write_match_json(
            opt,
            scene_proc.size(),
            templ_proc.size(),
            hits_topk,
            search_rois,
            roi_fallback_used,
            roi_source,
            err
        );
        if (json_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return json_code;
        }
    }

    cv::Mat vis;
    if (!to_bgr(scene, vis, err))
    {
        fmt::println(stderr, "{}", err);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    if (opt.draw_roi)
    {
        for (int i = 0; i < static_cast<int>(search_rois.size()); i++)
        {
            const auto &h = search_rois[i];
            cv::rectangle(vis, h, cv::Scalar(0, 0, 255), 1);
        }
    }
    for (int i = 0; i < static_cast<int>(hits_topk.size()); i++)
    {
        const auto &h = hits_topk[i];

        cv::rectangle(vis, h.bbox, cv::Scalar(0, 255, 0), opt.thickness);

        int y = (h.bbox.y > 5) ? (h.bbox.y - 5) : 0;
        cv::Point text_pos(h.bbox.x, y);

        if (opt.draw != "bbox")
        {
            const bool with_score = (opt.draw == "bbox+label+score");
            const auto text = with_score ? fmt::format("#{} conf={:.2f} s={:.2f}", i, h.confidence, h.scale)
                                         : fmt::format("#{} s={:.2f}", i, h.scale);

            cv::putText(
                vis,
                text,
                text_pos,
                cv::FONT_HERSHEY_SIMPLEX,
                opt.font_scale,
                cv::Scalar(0, 255, 0),
                1);
        }
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