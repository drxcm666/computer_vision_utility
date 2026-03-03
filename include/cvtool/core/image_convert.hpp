#pragma once 

#include <opencv2/core.hpp>

#include <string>

namespace cvtool::core::img
{
    cv::Mat to_gray(const cv::Mat &img);

    bool to_bgr(const cv::Mat &img, cv::Mat &dst, std::string &err);
}
