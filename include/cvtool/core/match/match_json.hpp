#pragma once 

#include "cvtool/core/exit_codes.hpp"
#include "cvtool/commands/match.hpp"
#include "cvtool/core/template_match.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace cvtool::core::match_json
{


cvtool::core::ExitCode write_match_json(
    const cvtool::cmd::MatchOptions &opt,
    const cv::Size &scene_size,
    const cv::Size &templ_size,
    const std::vector<cvtool::core::templ_match::MatchBest> &hits,
    const std::vector<cv::Rect> &rois,
    bool roi_fallback_used,
    const std::string &roi_source,
    std::string &err
);


}