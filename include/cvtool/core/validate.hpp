#pragma once

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/core/mat.hpp>

#include <string_view>
#include <string>

namespace cvtool::core::validate
{

cvtool::core::ExitCode validate_gray_channels(int channels, std::string &err);

cvtool::core::ExitCode validate_blur(const cv::Mat &img, int k, std::string &err);

cvtool::core::ExitCode validate_blur_k(int k, std::string &err);

cvtool::core::ExitCode validate_blur_fit(const cv::Mat &img, int k, std::string &err);

cvtool::core::ExitCode validate_thresholds(int low, int high, std::string &err);

cvtool::core::ExitCode validate_contours_thresh_mode(std::string_view mode, std::string &err);

cvtool::core::ExitCode validate_draw_mode(std::string_view draw, std::string &err);

cvtool::core::ExitCode validate_min_area(double min_area, std::string &err);

cvtool::core::ExitCode validate_adaptive_block(int block, std::string &err);

cvtool::core::ExitCode validate_manual_t(int t, std::string &err);

}
