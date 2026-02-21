#include "cvtool/commands/edges.hpp"
#include "cvtool/core/image_io.hpp"
#include "cvtool/core/edges_pipeline.hpp"

#include <opencv2/core/mat.hpp>

#include <fmt/core.h>

cvtool::core::ExitCode run_edges(const cvtool::cmd::EdgesOptions &opt)
{
    cv::Mat source_image;
    std::string err;

    const auto read_status = cvtool::core::image_io::read_image(opt.in_path, source_image, err);
    if (read_status != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return read_status;
    }

    cv::Mat edges_image;
    const auto pipeline_status = cvtool::core::edges_frame_to_gray(
        source_image,
        opt.threshold_low,
        opt.threshold_high,
        opt.blur_k,
        edges_image,
        err
    );
    if (pipeline_status != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return pipeline_status;
    }

    const auto write_status = cvtool::core::image_io::write_image(opt.out_path, edges_image, err);
    if (write_status != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return write_status;
    }

    fmt::println(
        "command: edges\n"
        "in: {}\n"
        "out: {}\n"
        "thresholds: {} - {}\n"
        "blur: {}\n"
        "status: ok",
        opt.in_path,
        opt.out_path,
        opt.threshold_low,
        opt.threshold_high,
        opt.blur_k);

    return cvtool::core::ExitCode::Ok;
}
