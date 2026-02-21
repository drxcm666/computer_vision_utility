#include "cvtool/core/video_io.hpp"

#include <fmt/format.h>

#include <filesystem>
#include <algorithm>
#include <cctype>

cvtool::core::ExitCode open_video_input(
    const std::string &in_path, 
    cv::VideoCapture &cap, 
    VideoMeta &meta, 
    std::string &err
)
{
    std::error_code ec;
    const std::filesystem::file_status st = std::filesystem::status(in_path, ec);

    if (ec)
    {
        err = fmt::format("error: cannot access input path: {} ({})", in_path, ec.message());
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }

    if (st.type() == std::filesystem::file_type::not_found)
    {
        err = fmt::format("error: input file not found: {}", in_path);
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }
    if (!std::filesystem::is_regular_file(st))
    {
        err = fmt::format("error: input is not a regular file: {}", in_path);
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }

    try
    {
        cap.open(in_path);
        if (!cap.isOpened())
        {
            err = fmt::format("error: cannot open video file: {}", in_path);
            return cvtool::core::ExitCode::CannotOpenOrReadInput;
        }
        
        meta.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        meta.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        meta.fps_in = cap.get(cv::CAP_PROP_FPS);
        meta.fps_out = meta.fps_in > 0 ? meta.fps_in : 30;

        if (meta.width <= 0 || meta.height <= 0)
        {
            err = fmt::format(
                "error: video has invalid resolution {}x{}: {}",
                meta.width,
                meta.height,
                in_path
            );
            return cvtool::core::ExitCode::CannotOpenOrReadInput;
        }

        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    catch(const cv::Exception &e)
    {
        err = fmt::format("error: cannot open video file: {} ({})", in_path, e.what());
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }
}

cvtool::core::ExitCode open_video_writer(
    const std::string &out_path, 
    const VideoMeta &meta, 
    std::string_view codec_req, 
    cv::VideoWriter &writer, 
    VideoMeta &meta_out,
    std::string &err
)
{
    if(out_path.empty())
    {
        err = fmt::format("error: output path is empty: {}", out_path);
        return cvtool::core::ExitCode::CannotOpenOutputVideo;
    }

    meta_out = meta;
    meta_out.codec_resolved.clear();

    auto parent_dir = std::filesystem::path(out_path).parent_path();

    if (!parent_dir.empty())
    {
        std::error_code ec;
        const bool parent_exist = std::filesystem::is_directory(parent_dir, ec);

        if (ec)
        {
            err = fmt::format("error: cannot access to parent directory: {} ({})", parent_dir.string(), ec.message());
            return cvtool::core::ExitCode::CannotOpenOutputVideo;
        }

        if (!parent_exist)
        {
            err = fmt::format("error: parent directory does not exist: {}", parent_dir.string());
            return cvtool::core::ExitCode::CannotOpenOutputVideo;
        }
    }

    std::string codec = std::string(codec_req);
    std::transform(codec.begin(), codec.end(), codec.begin(),
        [](unsigned char ch){ return static_cast<char>(std::tolower(ch)); });

    int fourcc{0};
    if (codec == "mp4v"){
        fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        meta_out.codec_resolved = "mp4v";
    }
    else if (codec == "mjpg"){ 
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        meta_out.codec_resolved = "mjpg";
    }
    else if (codec == "xvid"){ 
        fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        meta_out.codec_resolved = "xvid";
    }

    auto extension = std::filesystem::path(out_path).extension();
    std::string extension_str = extension.string();
    std::transform(
        extension_str.begin(),
        extension_str.end(),
        extension_str.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    if (codec == "auto")
    {
        if (extension_str == ".mp4") {
            fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            meta_out.codec_resolved = "mp4v (auto)";
        }
        else if (extension_str == ".avi") {
            fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            meta_out.codec_resolved = "xvid (auto)";
        }
        else {
            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            meta_out.codec_resolved = "mjpg (auto)";
        }
    }

    if (codec_req != "auto" && fourcc == 0)
    {
        err = fmt::format("error: invalid --codec: {}", codec_req);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    
    try
    {
        writer.open(out_path, fourcc, meta.fps_out, cv::Size(meta.width, meta.height), true);
        if (!writer.isOpened())
        {
            err = fmt::format(
                "error: cannot open output video: {}\n"
                "codec tried: {}\n"
                "hint: try --codec mjpg and/or output .avi",
                out_path, meta_out.codec_resolved);
            return cvtool::core::ExitCode::CannotOpenOutputVideo;
        }

        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    catch(const cv::Exception &e)
    {
        err = fmt::format("error: cannot open output video: {} ({})", out_path, e.what());
        return cvtool::core::ExitCode::CannotOpenOutputVideo;
    }
}
