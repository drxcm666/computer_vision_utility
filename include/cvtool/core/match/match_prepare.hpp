#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <string_view>

namespace cvtool::core::match_preparate
{

cvtool::core::ExitCode preparate_for_match(
    const cv::Mat &img, 
    std::string_view mode, 
    cv::Mat &out, 
    std::string &err
);

}