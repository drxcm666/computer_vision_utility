#include "cvtool/commands/contours.hpp"
#include "cvtool/core/image_io.hpp"
#include "cvtool/core/exit_codes.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/threshold.hpp"
#include "cvtool/core/contours_core.hpp"

#include <nlohmann/json.hpp>

#include <fmt/core.h>

#include <fstream>
#include <iomanip>
#include <algorithm>

cvtool::core::ExitCode run_contours(const cvtool::cmd::ContoursOptions &opt)
{
    cv::Mat img;
    std::string err;

    const auto blur_code = cvtool::core::validate::validate_blur_k(opt.blur_k, err);
    if (blur_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return blur_code;
    }

    const auto area_code = cvtool::core::validate::validate_min_area(opt.min_area, err);
    if (area_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return area_code;
    }

    const auto thresh_code = cvtool::core::validate::validate_contours_thresh_mode(opt.thresh, err);
    if (thresh_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return thresh_code;
    }

    if (opt.thresh == "adaptive")
    {
        const auto adaptive_code = cvtool::core::validate::validate_adaptive_block(opt.block, err);
        if (adaptive_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return adaptive_code;
        }
    }
    else if (opt.thresh == "manual")
    {
        const auto manual_code = cvtool::core::validate::validate_manual_t(opt.t, err);
        if (manual_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return manual_code;
        }
    }

    const auto draw_code = cvtool::core::validate::validate_draw_mode(opt.draw, err);
    if (draw_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return draw_code;
    }

    fmt::println(
        "command: contours\n"
        "in: {}\n"
        "out: {}\n"
        "thresh: {}\n"
        "params: blur_k={} min_area={} invert={} draw={}",
        opt.in_path,
        opt.out_path,
        opt.thresh,
        opt.blur_k, opt.min_area, opt.invert, opt.draw);

    const cvtool::core::ExitCode read_code = cvtool::core::image_io::read_image(opt.in_path, img, err);
    if (read_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return read_code;
    }

    cv::Mat bin;
    const cvtool::core::ExitCode mask_code = make_binary_mask(
        img,
        opt.thresh,
        opt.blur_k,
        opt.invert,
        opt.block,
        opt.c,
        opt.t,
        bin,
        err);
    if (mask_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return mask_code;
    }

    std::vector<cvtool::core::contours::ContourItem> items;
    cvtool::core::contours::ContourStats stats;
    const cvtool::core::ExitCode cont_code = cvtool::core::contours::find_contours_report(bin, opt.min_area, items, stats, err);
    if (cont_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return cont_code;
    }

    cv::Mat annotated = img.clone();
    if (annotated.channels() == 1)
    {
        cv::cvtColor(annotated, annotated, cv::COLOR_GRAY2BGR);
    }
    else if (annotated.channels() == 4)
    {
        cv::cvtColor(annotated, annotated, cv::COLOR_BGRA2BGR);
    }

    for (const auto &i : items)
    { 
        if (opt.draw == "bbox" || opt.draw == "both")
        {
            cv::rectangle(annotated, i.bbox, cv::Scalar(0, 255, 0), 2);
        }
        if (opt.draw == "contour" || opt.draw == "both")
        {
            std::vector<std::vector<cv::Point>> one{i.contour};
            cv::drawContours(annotated, one, -1, cv::Scalar(255, 0, 0), 2);
        }

        int y = (i.bbox.y > 5) ? (i.bbox.y - 5) : 0;
        cv::Point text_pos(i.bbox.x, y);

        cv::putText(
            annotated, 
            fmt::format("#{} area:{:.1f}", i.id, i.area), 
            text_pos, 
            cv::FONT_HERSHEY_SIMPLEX, 
            0.5, 
            cv::Scalar(0, 255, 0), 
            1
        );
    }    

    if(!opt.json_path.empty())
    {
        static constexpr size_t kMaxItems = 200;
        bool truncated = items.size() > kMaxItems;
        std::size_t n = std::min(items.size(), kMaxItems);

        nlohmann::ordered_json items_array = nlohmann::ordered_json::array();
        for (std::size_t idx = 0; idx < n; idx++)
        {
            const auto &i = items[idx];
            nlohmann::ordered_json single_item;
            single_item["id"] = i.id;
            single_item["area"] = i.area;
            single_item["bbox"] = {
                {"x", i.bbox.x},
                {"y", i.bbox.y},
                {"w", i.bbox.width},
                {"h", i.bbox.height}
            };
            items_array.push_back(single_item);
        }

        nlohmann::ordered_json j;
        j["command"] = "contours";
        j["input"] = opt.in_path;
        j["output"] = opt.out_path;
        j["threshold"]["mode"] = opt.thresh;
        j["threshold"]["blur_k"] = opt.blur_k;
        j["threshold"]["invert"] = opt.invert;
        if (opt.thresh == "adaptive")
        {
            j["threshold"]["params"] = {
                {"block", opt.block},
                {"c", opt.c}
            };
        } else if (opt.thresh == "manual")
        {
            j["threshold"]["params"] = {
                {"t", opt.t}
            };
        } else
        {
            j["threshold"]["params"] = nlohmann::ordered_json::object();
        }
        
        j["stats"] = {
            {"contours_total", stats.contours_total},
            {"contours_kept", stats.contours_kept},
            {"area_min", stats.area_min},
            {"area_mean", stats.area_mean},
            {"area_max", stats.area_max}
        };
        j["items_truncated"] = truncated;
        j["items"] = items_array;
        

        std::ofstream file(opt.json_path);
        if (!file)
        {
            fmt::println(stderr, "error: cannot open json output '{}'", opt.json_path);
            return cvtool::core::ExitCode::CannotWriteOutput;
        }

        file << std::setw(4) << j << "\n";
        if (!file.good())
        {
            fmt::println(stderr, "error: failed to write json output: {}", opt.json_path);
            return cvtool::core::ExitCode::CannotWriteOutput;
        }
    }

    const cvtool::core::ExitCode write_code = cvtool::core::image_io::write_image(opt.out_path, annotated, err);
    if (write_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return write_code;
    }

    fmt::println(
        "status: ok\n"
        "contours_total: {}\n"
        "contours_kept: {}\n"
        "area_min: {}\n"
        "area_mean: {}\n"
        "area_max: {}",
        stats.contours_total,
        stats.contours_kept,
        stats.area_min,
        stats.area_mean,
        stats.area_max
    );

    return cvtool::core::ExitCode::Ok;
}