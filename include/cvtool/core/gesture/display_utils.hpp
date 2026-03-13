#pragma once 

#include <opencv2/opencv.hpp>

namespace cvtool::core::gesture
{

cv::Mat letterbox(const cv::Mat &source, int targetWidth, int targetHeight);

}