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
    std::string &err
)
{
    nlohmann::ordered_json j;


    j["command"] = "match";
    j["input"] = opt.in_path;
    j["template"] = opt.templ_path;
    j["output"] = opt.out_path;
    j["params"] = {{"mode", opt.mode}, {"method", opt.method},
                {"max_results", opt.max_results}, {"min_score", opt.min_score}, {"nms", opt.nms},
                {"draw", opt.draw}, {"thickness", opt.thickness}, {"font_scale", opt.font_scale}};
    j["template_size"] = {{"w", templ_size.width}, {"h", templ_size.height}};
    j["scene_size"] = {{"w", scene_size.width}, {"h", scene_size.height}};

    if (!opt.roi.empty()) {
        cv::Rect r;
        std::string err_dummy;
        if (cvtool::core::validate::validate_roi(opt.roi, r, err_dummy) == cvtool::core::ExitCode::Ok) {
            j["params"]["roi"] = { {"x", r.x}, {"y", r.y}, {"w", r.width}, {"h", r.height} };
        } else {
            j["params"]["roi"] = nlohmann::ordered_json::object();
        }
    } else {
        j["params"]["roi"] = nlohmann::ordered_json::object();
    }


    j["matches"] = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(hits.size()); ++i) {
        const auto& h = hits[i];
        j["matches"].push_back({
            {"id", i},
            {"bbox", {{"x", h.bbox.x}, {"y", h.bbox.y}, {"w", h.bbox.width}, {"h", h.bbox.height}}},
            {"raw_score", h.raw_score},
            {"confidence", h.confidence}
        });
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
 
    cv::Mat result;
    cv::Mat *out_res = nullptr;
    if (!opt.heatmap_path.empty())
    {
        out_res = &result;
    }

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
    const cv::Mat scene_search = !opt.roi.empty() ? scene_proc(roi) : scene_proc;

    if (templ_proc.cols > scene_search.cols || templ_proc.rows > scene_search.rows)
    {
        const auto scene_w = scene_search.cols;
        const auto scene_h = scene_search.rows;
        fmt::println(stderr, "error: template larger than scene (templ: {}x{}, scene: {}x{})",
                     templ_proc.cols, templ_proc.rows, scene_w, scene_h);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    const int candidates = opt.max_results * 10;
    const auto cands = cvtool::core::templ_match::match_topk(scene_search, templ_proc, method, candidates, opt.min_score, out_res);
    auto hits_topk = cvtool::core::templ_match::nms_iou(cands, opt.nms, opt.max_results);

    
    if (!opt.roi.empty())
    {
        for (auto &h : hits_topk)
        {
            h.bbox.x += roi.x;
            h.bbox.y += roi.y;
        }
    }
    if(out_res)
    {
        auto heat_code = make_heatmap(result, method, opt.heatmap_path, err);
        if (heat_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return heat_code;
        }
    }
    if (!opt.json_path.empty())
    {
        const auto json_code = write_match_json(opt, scene_proc.size(), templ_proc.size(), hits_topk, err);
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
    for (int i = 0; i <static_cast<int>(hits_topk.size()); i++)
    {
        const auto &h = hits_topk[i];

        cv::rectangle(vis, h.bbox, cv::Scalar(0, 255, 0), opt.thickness);

        int y = (h.bbox.y > 5) ? (h.bbox.y - 5) : 0;
        cv::Point text_pos(h.bbox.x, y);

        if (opt.draw != "bbox")
        {
            const bool with_score = (opt.draw == "bbox+label+score");
            const auto text = with_score ? fmt::format("#{} conf:{:.2f}", i, h.confidence)
                                         : fmt::format("#{}", i);

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
        fmt::println("best: conf={:.2f} raw={:.4f} at x={} y={}",
                     hits_topk[0].confidence, hits_topk[0].raw_score, hits_topk[0].bbox.x, hits_topk[0].bbox.y);


    const auto write_code = cvtool::core::image_io::write_image(opt.out_path, vis, err);
    if (write_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return write_code;
    }

    return cvtool::core::ExitCode::Ok;
}