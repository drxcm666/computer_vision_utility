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
};

}

cvtool::core::ExitCode run_match(const cvtool::cmd::MatchOptions &opt);
