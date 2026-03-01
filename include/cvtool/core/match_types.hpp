#pragma once 

#include "cvtool/commands/match.hpp"
#include "cvtool/core/template_match.hpp"

#include <opencv2/core.hpp>

#include <vector>
#include <string>

namespace cvtool::core::match
{

struct MatchContext
{
    const cvtool::cmd::MatchOptions *opt{nullptr};

    cv::Mat scene_proc;
    cv::Mat templ_proc;

    int method{0};

    double scale_min{1.0}, scale_max{1.0}, scale_step{1.0};
    int count{1};
    int per_scale_top{0};
};

struct RoiInfo
{
    std::vector<cv::Rect> search_rois;
    bool roi_fall_back_used{false};
    std::string roi_source;
};

struct MatchArtifacts
{
    std::vector<cvtool::core::templ_match::MatchBest> hits_topk;
    cv::Mat best_result;
};

    
}