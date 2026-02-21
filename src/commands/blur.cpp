#include "cvtool/commands/blur.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/image_io.hpp"

#include <opencv2/imgproc.hpp>

#include <fmt/format.h>


cvtool::core::ExitCode run_blur(const cvtool::cmd::BlurOptions &opt)
{
    cv::Mat img;
    std::string err;

    auto fail = [&](cvtool::core::ExitCode c) {
        fmt::println(stderr, "{}", err);
        return c;
    };

    const auto read_code = cvtool::core::image_io::read_image(opt.in_path, img, err);

    if (read_code != cvtool::core::ExitCode::Ok)
    {
        return fail(read_code);
    }

    err.clear();
    const auto blur_code = cvtool::core::validate::validate_blur(img, opt.blur_k, err);
    if (blur_code != cvtool::core::ExitCode::Ok)
    {
        return fail(blur_code);
    }

    if (opt.blur_k >= 3)
    {
        try
        {
            cv::Mat out = img.clone();
            cv::GaussianBlur(img, img, cv::Size(opt.blur_k, opt.blur_k), 0, 0);
        }
        catch (const cv::Exception &e)
        {
            fmt::println(stderr, "error: blur failed ({})", e.what());
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
    }

    err.clear();
    const auto write_code = cvtool::core::image_io::write_image(opt.out_path, img, err);
    if (write_code != cvtool::core::ExitCode::Ok)
    {
        return fail(write_code);
    }

    fmt::println(
        "command: blur\n"
        "in: {}\n"
        "out: {}\n"
        "blur_k: {}\n"
        "status: ok",
        opt.in_path,
        opt.out_path,
        opt.blur_k);

    return cvtool::core::ExitCode::Ok;
}
