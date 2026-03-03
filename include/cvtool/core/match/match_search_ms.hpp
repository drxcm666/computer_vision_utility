#pragma once 

#include "cvtool/core/exit_codes.hpp"
#include "cvtool/core/template_match.hpp"

#include <opencv2/imgproc.hpp>

#include <vector>

namespace cvtool::core::match_search_ms
{

cvtool::core::ExitCode search_multiscale(
    const cv::Mat &scene_proc,
    const cv::Mat &templ_proc,
    int method,
    const std::vector<cv::Rect> &search_rois,
    double scale_min,
    double scale_step,
    int count,
    int per_scale_top,
    double min_score,
    bool need_heatmap,
    std::vector<cvtool::core::templ_match::MatchBest> &out_all,
    cv::Mat &out_best_result,
    int &out_valid_scales,
    std::string &err
);


}