#pragma once 

#include "cvtool/core/exit_codes.hpp"
#include "cvtool/core/template_match.hpp"

#include <opencv2/imgproc.hpp>

#include <string_view>

namespace cvtool::core::match_render 
{

cvtool::core::ExitCode render (
    const std::vector<cv::Rect> &rois,
    const std::vector<cvtool::core::templ_match::MatchBest> &hits_topk,
    const cv::Mat &scene,
    bool draw_roi,
    std::string_view draw_mode,
    int thickness,
    double font_scale,
    cv::Mat &vis,
    std::string &err
);



}