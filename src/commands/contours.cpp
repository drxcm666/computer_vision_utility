#include "cvtool/commands/contours.hpp"
#include "cvtool/core/image_io.hpp"
#include "cvtool/core/exit_codes.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/threshold.hpp"
#include "cvtool/core/contours_core.hpp"

#include <fmt/core.h>

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
        opt.blur_k, opt.min_area, opt.invert, opt.draw
    );

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
        err
    );
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

    fmt::println("contours_total: {}", stats.contours_total);
    fmt::println("contours_kept: {}", stats.contours_kept);
    fmt::println("area_min: {}", stats.area_min);
    fmt::println("area_mean: {}", stats.area_mean);
    fmt::println("area_max: {}", stats.area_max);

    
    const cvtool::core::ExitCode write_code = cvtool::core::image_io::write_image(opt.out_path, bin, err);
    if (write_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return write_code;
    }

    fmt::println("status: ok");

    return cvtool::core::ExitCode::Ok;
}