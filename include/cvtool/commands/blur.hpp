#pragma once

#include "cvtool/core/exit_codes.hpp"

#include <string>

namespace cvtool::cmd {

struct BlurOptions
{
    std::string in_path, out_path;
    int blur_k{0};
};

}

cvtool::core::ExitCode run_blur(const cvtool::cmd::BlurOptions &opt);
