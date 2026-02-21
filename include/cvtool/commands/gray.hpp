#pragma once

#include "cvtool/core/exit_codes.hpp"

#include <string>

namespace cvtool::cmd {

struct GrayOptions { 
    std::string in_path; 
    std::string out_path; 
};

}

cvtool::core::ExitCode run_gray(const cvtool::cmd::GrayOptions &opt);
