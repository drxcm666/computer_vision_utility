#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <string>

namespace cvtool::cmd
{

struct ContoursOptions {
    std::string in_path;
    std::string out_path;
    std::string thresh{"otsu"};
    int blur_k{0};
    double min_area{100.0};
    std::string draw{"bbox"};
    bool invert{false};

    int block{11};
    double c{2.0};

    int t{-1};

    std::string json_path{};
};

}

cvtool::core::ExitCode run_contours(const cvtool::cmd::ContoursOptions &opt);
