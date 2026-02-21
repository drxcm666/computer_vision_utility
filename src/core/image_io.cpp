#include "cvtool/core/image_io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <fmt/format.h>
#include <filesystem>

namespace cvtool::core::image_io
{

cvtool::core::ExitCode read_image(const std::string &in_path, cv::Mat &out_image, std::string &err)
{
    err.clear();

    if (in_path.empty())
    {
        err = "error: input path is empty";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    std::error_code ec;
    const bool exists = std::filesystem::is_regular_file(in_path, ec);

    if (ec)
    {
        err = fmt::format("error: cannot access input path: {} ({})", in_path, ec.message());
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }
    if (!exists)
    {
        err = fmt::format("error: input file not found: {}", in_path);
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }
    
    try
    {
        cv::Mat img = cv::imread(in_path, cv::IMREAD_UNCHANGED);

        if (img.empty())
        {
            err = fmt::format("error: cannot read image: {}", in_path);
            return cvtool::core::ExitCode::CannotOpenOrReadInput;
        }

        out_image = img;
        err.clear();

        return cvtool::core::ExitCode::Ok;
    }
    catch(const cv::Exception &e)
    {
        err = fmt::format("error: cannot read image: {} ({})", in_path, e.what());
        return ExitCode::CannotOpenOrReadInput;
    }
}

cvtool::core::ExitCode write_image(const std::string &out_path, const cv::Mat &out_image, std::string &err)
{
    err.clear();

    if (out_path.empty())
    {
        err = "error: output path is empty";
        return cvtool::core::ExitCode::CannotWriteOutput;
    }

    auto parent_dir = std::filesystem::path(out_path).parent_path();
    
    if (!parent_dir.empty())
    {
        std::error_code ec;
        const auto dir_status = std::filesystem::status(parent_dir, ec);
        if (ec)
        {
            err = fmt::format("error: cannot access parent path: {} ({})", parent_dir.string(), ec.message());
            return cvtool::core::ExitCode::CannotWriteOutput;
        }

        if (!std::filesystem::exists(dir_status))
        {
            err = fmt::format("error: parent directory does not exist: {}", parent_dir.string());
            return cvtool::core::ExitCode::CannotWriteOutput;
        }

        if (!std::filesystem::is_directory(dir_status))
        {
            err = fmt::format("error: parent path is not a directory: {}", parent_dir.string());
            return cvtool::core::ExitCode::CannotWriteOutput;
        }
    }

    if (out_image.empty())
    {
        err = "error: output image is empty";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    try
    {
        if(!cv::imwrite(out_path, out_image))
        {
            err = fmt::format("error: cannot write image to: {}", out_path);
            return cvtool::core::ExitCode::CannotWriteOutput;
        }

        err.clear();

        return cvtool::core::ExitCode::Ok;
    }
    catch(const cv::Exception &e)
    {
        err = fmt::format("error: cannot write image to: {} ({})", out_path, e.what());
        return ExitCode::CannotWriteOutput;
    }
}

}
