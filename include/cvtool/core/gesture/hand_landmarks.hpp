#pragma once 

#include <opencv2/opencv.hpp>

#include <array>

namespace cvtool::core::gesture{

enum class Handedness 
{
    Right,
    Left,
    None
};

struct HandLandmarkResult
{
    bool has_hand{false};
    float confidence{0.0f};
    std::array<cv::Point2f, 21> points;
    Handedness hand{Handedness::None};
    cv::Rect hand_bbox{};
};


}