#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace cvtool::core::match_heatmap
{

cvtool::core::ExitCode write_heatmap(
    const cv::Mat &result,
    int method,
    const std::string &out_path,
    std::string &err);

}