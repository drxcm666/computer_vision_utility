#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <string>

#include <opencv2/core/mat.hpp>

cvtool::core::ExitCode make_binary_mask(
    const cv::Mat &src, 
    const std::string &mode, 
    int blur_k, bool invert, int block, double c, int t, 
    cv::Mat &out_bin, std::string &err
);