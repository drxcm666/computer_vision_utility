#pragma once 

#include "cvtool/core/exit_codes.hpp"
#include "cvtool/core/match_types.hpp"

namespace cvtool::core::roi_edges
{

struct RoiEdgesParams
{
    int low{60};
    int high{140};
    int blur_k{5};

    int roi_max{8};
    double min_area{0.01};
    int pad{10};
    double merge_iou{20};
};

std::vector<cv::Rect> build_rois_edges(const cv::Mat &scene_gray, RoiEdgesParams &p);


}