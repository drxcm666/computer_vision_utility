#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <string>

namespace cvtool::cmd {

struct MatchOptions
{
    std::string in_path;
    std::string out_path;
    std::string templ_path;
    double min_score{0.80};
    std::string method{"ccoeff_normed"};
    int max_results{5};
    double nms{0.30};
    std::string mode{"gray"};
    std::string heatmap_path;
    std::string json_path;
    
    std::string roi{};
    std::string draw{"bbox+label+score"};
    int thickness{2};
    double font_scale{0.5};
    std::string scales;
    int per_scale_top{0};
    int max_scales{0};

    std::string roi_auto{};
    int roi_max{8};
    double roi_min_area{0.01};
    int roi_pad{10};
    double roi_merge_iou{0.20};
    bool roi_fallback{true};
    int roi_edges_low{60};
    int roi_edges_high{140};
    int roi_edges_blur_k{5};
    bool draw_roi{false};
};

}

cvtool::core::ExitCode run_match(const cvtool::cmd::MatchOptions &opt);
