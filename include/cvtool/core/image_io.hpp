#pragma once

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/core/mat.hpp>

#include <string>

namespace cvtool::core::image_io
{

cvtool::core::ExitCode read_image(const std::string &in_path, cv::Mat &out_image, std::string &err);

cvtool::core::ExitCode write_image(const std::string &out_path, const cv::Mat &out_image, std::string &err);

}
