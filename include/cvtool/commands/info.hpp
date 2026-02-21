#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <string>

namespace cvtool::cmd {

struct InfoOptions { std::string in_path; };

}

cvtool::core::ExitCode run_info(const cvtool::cmd::InfoOptions &opt);
