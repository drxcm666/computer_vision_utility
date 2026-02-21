#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <string>

namespace cvtool::cmd
{

struct EdgesOptions {
    std::string in_path, out_path;
    int blur_k{0};
    int threshold_low{0};
    int threshold_high{0};
};

}

cvtool::core::ExitCode run_edges(const cvtool::cmd::EdgesOptions &opt);
