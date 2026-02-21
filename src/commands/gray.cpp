#include "cvtool/commands/gray.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/image_io.hpp"

#include <opencv2/imgproc.hpp>

#include <fmt/format.h>


cvtool::core::ExitCode run_gray(const cvtool::cmd::GrayOptions &opt)
{
    cv::Mat img;
    std::string err;
    const cvtool::core::ExitCode img_code = cvtool::core::image_io::read_image(opt.in_path, img, err);

    if (img_code != cvtool::core::ExitCode::Ok){
        fmt::println(stderr, "{}", err);
        return img_code;
    }
    
    const int channels = img.channels(); 
    const auto v = cvtool::core::validate::validate_gray_channels(channels, err);
    if (v != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return v;
    }

    if (img.depth() != CV_8U)
    {
        fmt::println(stderr, "error: only 8-bit images are supported for this command");
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    

    cv::Mat gray;
    try
    {
        if (channels == 1){
            gray = img.clone();
        } else if (channels == 3){
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else if (channels == 4){
            cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
        }
    }
    catch (const cv::Exception &e)
    {
        fmt::println(stderr, "error: grayscale conversion failed ({})", e.what());
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    if (gray.empty() || gray.channels() != 1){
        fmt::println(stderr, "error: grayscale conversion failed");
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    
    const cvtool::core::ExitCode write_code = cvtool::core::image_io::write_image(opt.out_path, gray, err);
    if (write_code != cvtool::core::ExitCode::Ok){
        fmt::println(stderr, "{}", err);
        return write_code;
    }

    fmt::println(
        "command: gray\n"
        "in: {}\n"
        "out: {}\n"
        "status: ok",
        opt.in_path, 
        opt.out_path);

    return cvtool::core::ExitCode::Ok;
}
