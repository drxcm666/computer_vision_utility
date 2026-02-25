#pragma once

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/core.hpp>

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

cvtool::core::ExitCode validate_01(std::string_view name, double v, std::string &err);

cvtool::core::ExitCode  validate_max_results(int n, std::string &err);

cvtool::core::ExitCode validate_mode_match(std::string_view mode, std::string &err);

cvtool::core::ExitCode validate_method_match(std::string_view method_str, int &method_out, std::string &err);

cvtool::core::ExitCode validate_roi(std::string_view str, cv::Rect &out, std::string &err);

cvtool::core::ExitCode validate_draw_match(std::string_view draw, std::string &err);

cvtool::core::ExitCode validate_thickness(int thickness, std::string &err);

cvtool::core::ExitCode validate_font_scale(double fs, std::string &err);


}
