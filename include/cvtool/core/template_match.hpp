#pragma once

#include <opencv2/core.hpp>

#include <vector> 

namespace cvtool::core::templ_match 
{

struct MatchBest
{
    cv::Rect bbox;
    double raw_score{};
    double confidence{};
};

MatchBest match_best(const cv::Mat &scene, const cv::Mat &templ, int method);

std::vector<MatchBest> match_topk(
    const cv::Mat &scene,
    const cv::Mat &templ,
    int method, 
    int max_results,
    double min_score,
    cv::Mat *out_result = nullptr
);

std::vector<MatchBest> nms_iou(
    const std::vector<MatchBest> &hits,
    double iou_thr,
    int max_keep
);



}