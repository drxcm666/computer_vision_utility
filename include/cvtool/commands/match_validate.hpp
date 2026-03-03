#pragma once 

#include "cvtool/core/exit_codes.hpp"
#include "cvtool/commands/match.hpp"

#include <string>

namespace cvtool::cmd::match_validate
{

cvtool::core::ExitCode validate_match_options(
    const cvtool::cmd::MatchOptions &opt,
    int &method,
    double &scale_min, double &scale_max, double &scale_step,
    int &count, int &per_scale_top,
    std::string &err

);
    
}