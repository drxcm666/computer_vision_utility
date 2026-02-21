#include "cvtool/commands/video_edges.hpp"
#include "cvtool/core/edges_pipeline.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/video_io.hpp"

#include <fmt/format.h>

#include <string_view>
#include <chrono>

bool is_allowed_codec(std::string_view codec)
{
    return codec == "auto" || codec == "mp4v" || codec == "mjpg" || codec == "xvid";
}

cvtool::core::ExitCode run_video_edges(const cvtool::cmd::VideoEdgesOptions &opt)
{
    std::string err;
    const auto threshold_status = cvtool::core::validate::validate_thresholds(opt.low, opt.high, err);

    if (threshold_status != cvtool::core::ExitCode::Ok)
        err = err.empty() ? "error: invalid thresholds (require 0<=low<high<=255)" : err;

    else if (cvtool::core::validate::validate_blur_k(opt.blur_k, err) != cvtool::core::ExitCode::Ok)
        err = err.empty() ? "error: invalid --blur-k (must be 0 or odd >= 3)" : err;
    
    else if (opt.every < 1)
        err = "error: invalid --every (must be >= 1)";

    else if (opt.max_frames < 0)
        err = "error: invalid --max-frames (must be >= 0)";
    
    else if (!is_allowed_codec(opt.codec))
        err = "error: invalid --codec (allowed: auto, mp4v, mjpg, xvid)";


    if (!err.empty())
    {
        fmt::println(stderr, "{}", err);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    cv::VideoCapture cap;
    VideoMeta meta;
    const auto open_code = open_video_input(opt.in_path, cap, meta, err);
    if (open_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return open_code;
    }

    if (meta.width <= 0 || meta.height <= 0)
    {
        fmt::println(stderr, "error: invalid video size: {}x{}", meta.width, meta.height);
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }

    if (opt.every > 1) {
        meta.fps_out = meta.fps_out / opt.every;
    }

    cv::VideoWriter writer;
    VideoMeta meta_out;
    const auto writer_code = open_video_writer(opt.out_path, meta, opt.codec, writer, meta_out, err);
    if (writer_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return writer_code;
    }

    fmt::println(
        "command: video-edges\n"
        "in: {}\n"
        "out: {}\n"
        "size: {}x{}\n"
        "fps_in: {}\n"
        "fps_out: {:.2f}\n"
        "codec: {}\n"
        "params: low={} high={} blur_k={} every={} max_frames={}",
        opt.in_path,
        opt.out_path,
        meta.width,
        meta.height,
        (meta.fps_in > 0 ? fmt::format("{:.2f}", meta.fps_in) : "unknown"),
        meta.fps_out,
        meta_out.codec_resolved,
        opt.low,
        opt.high,
        opt.blur_k,
        opt.every,
        opt.max_frames);

    cv::Mat frame, out;
    int frames_read{0}, frames_written{0}, frames_processed{0};
    auto t0 = std::chrono::steady_clock::now();

    try
    {
        while (cap.read(frame))
        {
            frames_read++;

            if ((frames_read - 1) % opt.every != 0)
                continue;

            if (opt.max_frames > 0 && frames_processed >= opt.max_frames)
                break;

            auto rc = cvtool::core::edges_frame_to_bgr(frame, opt.low, opt.high, opt.blur_k, out, err);
            if (rc != cvtool::core::ExitCode::Ok)
            {
                fmt::println(stderr, "{}", err);
                return rc;
            }

            frames_processed++;

            try
            {
                writer.write(out);
                frames_written++;
            }
            catch(const cv::Exception &e)
            {
                fmt::println(stderr, "error: failed to write frame {} ({})", frames_processed, e.what());
                return cvtool::core::ExitCode::CannotOpenOutputVideo;
            }

            if (frames_processed % 30 == 0)
            {
                fmt::println(
                    "progress: read={}, processed={}, written={}",
                    frames_read,
                    frames_processed,
                    frames_written);
            }
        }
    }
    catch (const cv::Exception &e)
    {
        fmt::println(stderr, "error: video processing failed ({})", e.what());
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }

    if (frames_read == 0)
    {
        fmt::println(stderr, "error: cannot read frames from video: {}", opt.in_path);
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);

    std::chrono::milliseconds avg_ms_per_frame{};
    if (frames_processed > 0)
        avg_ms_per_frame = ms / frames_processed;

    fmt::println(
        "status: ok\n"
        "read: {}\n"
        "processed: {}\n"
        "written: {}\n"
        "time_ms: {}\n"
        "avg_ms_per_frame: {}",
        frames_read,
        frames_processed,
        frames_written,
        ms.count(),
        avg_ms_per_frame.count());

    return cvtool::core::ExitCode::Ok;
}
