#pragma once

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/core/mat.hpp>

#include <string>

namespace cvtool::core
{

cvtool::core::ExitCode edges_frame_to_bgr(
    const cv::Mat &frame,
    int low, 
    int high, 
    int blur_k, 
    cv::Mat &out_bgr,
    std::string &err 
);

cvtool::core::ExitCode edges_frame_to_gray(
    const cv::Mat &frame,
    int low,
    int high,
    int blur_k,
    cv::Mat &out_gray,
    std::string &err
);

}
