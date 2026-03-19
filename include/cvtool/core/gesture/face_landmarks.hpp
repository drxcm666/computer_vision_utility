#pragma once 

#include <opencv2/opencv.hpp>


namespace cvtool::core::gesture
{

struct FaceLandmarkResult
{
    bool has_face{false};
    float confidence{0.0f};
    cv::Rect bbox;
    cv::Point2f mouth_center{0.0f, 0.0f};
    cv::Point2f mouth_left{0.0f, 0.0f};
    cv::Point2f mouth_right{0.0f, 0.0f};
};


}