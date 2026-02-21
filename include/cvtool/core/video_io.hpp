#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/videoio.hpp>

#include <string_view>
#include <string>

struct VideoMeta
{
    int width{};
    int height{};
    double fps_in{};
    double fps_out{};
    std::string codec_resolved;
};

cvtool::core::ExitCode open_video_input(const std::string &in_path, cv::VideoCapture &cap, VideoMeta &meta, std::string &err);

cvtool::core::ExitCode open_video_writer(
    const std::string &out_path, 
    const VideoMeta &meta, 
    std::string_view codec_req, 
    cv::VideoWriter &writer, 
    VideoMeta &meta_out,
    std::string &err
);
