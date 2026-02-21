#include "cvtool/commands/info.hpp"
#include "cvtool/core/exit_codes.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

#include <fmt/format.h>

#include <filesystem>


static void print_image_info(const std::filesystem::path &path, const cv::Mat &img)
{
    const char *depth_str = cv::depthToString(img.depth());
    const std::string type_str = cv::typeToString(img.type());

    fmt::println(
        "kind: image\n"
        "path: {}\n"
        "size: {}x{}\n"
        "channels: {}\n"
        "depth: {} ({})\n"
        "mat_type: {}",
        path.string(),
        img.cols,
        img.rows,
        img.channels(),
        depth_str,
        img.depth(),
        type_str);
}

static cvtool::core::ExitCode print_video_info(const std::filesystem::path &path, cv::VideoCapture &cap)
{
    const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    const bool fps_known = fps > 0;
    const bool frames_known = frames > 0;
    const bool duration_known = fps_known && frames_known;

    const std::string fps_str = fps_known ? fmt::format("{:.2f}", fps) : "unknown";
    const std::string frames_str =
        frames_known ? fmt::format("{}", static_cast<long long>(frames)) : "unknown";
    const std::string duration_str =
        duration_known ? fmt::format("{:.2f}", (frames / fps)) : "unknown";

    fmt::println(
        "kind: video\n"
        "path: {}\n"
        "size: {}x{}\n"
        "fps: {}\n"
        "frames: {}\n"
        "duration_s: {}",
        path.string(),
        w,
        h,
        fps_str,
        frames_str,
        duration_str);

    return cvtool::core::ExitCode::Ok;
}


cvtool::core::ExitCode run_info(const cvtool::cmd::InfoOptions &opt)
{
    std::error_code ec;
    const std::filesystem::file_status st = std::filesystem::status(opt.in_path, ec);

    if (ec)
    {
        fmt::println(stderr, "error: cannot access input path: {} ({})", opt.in_path, ec.message());
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }

    if (st.type() == std::filesystem::file_type::not_found)
    {
        fmt::println(stderr, "error: input file not found: {}", opt.in_path);
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }
    if (!std::filesystem::is_regular_file(st))
    {
        fmt::println(stderr, "error: input is not a regular file: {}", opt.in_path);
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }

    try
    {
        cv::Mat img = cv::imread(opt.in_path, cv::IMREAD_UNCHANGED);
        if (!img.empty())
        {
            print_image_info(opt.in_path, img);
            return cvtool::core::ExitCode::Ok;
        }
    }
    catch (const cv::Exception &e)
    {
        fmt::println(stderr, "error: cannot read image: {} ({})", opt.in_path, e.what());
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }

    try
    {
        cv::VideoCapture cap(opt.in_path);
        if (cap.isOpened())
        {
            return print_video_info(opt.in_path, cap);
        }
    }
    catch (const cv::Exception &e)
    {
        fmt::println(stderr, "error: cannot read video: {} ({})", opt.in_path, e.what());
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }

    fmt::println(stderr, "error: unsupported or corrupted media: {}", opt.in_path);
    return cvtool::core::ExitCode::CannotOpenOrReadInput;
}
