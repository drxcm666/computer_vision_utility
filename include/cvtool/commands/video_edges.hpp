#pragma once

#include "cvtool/core/exit_codes.hpp"
#include <string>

namespace cvtool::cmd
{

struct VideoEdgesOptions
{
    std::string in_path;
    std::string out_path;
    int low{};
    int high{};
    int blur_k{0};
    int every{1};
    int max_frames{0};
    std::string codec{"auto"}; 
};

}

cvtool::core::ExitCode run_video_edges(const cvtool::cmd::VideoEdgesOptions &opt);
