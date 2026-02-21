#include "cvtool/core/validate.hpp"

#include <fmt/format.h>
#include <algorithm>

namespace cvtool::core::validate
{

cvtool::core::ExitCode validate_gray_channels(int channels, std::string &err)
{
    if (channels == 1 || channels == 3 || channels == 4) 
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }

    err = fmt::format("error: unsupported channel count: {}", channels);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_blur(const cv::Mat &img, int k, std::string &err)
{
    // composite: parameter rules + fit-in-image
    const cvtool::core::ExitCode p = validate_blur_k(k, err);
    if (p != cvtool::core::ExitCode::Ok) return p;
    return validate_blur_fit(img, k, err);
}

cvtool::core::ExitCode validate_blur_k(int k, std::string &err)
{
    // allow 0 (no-op) or odd >= 3
    if (k == 0 || (k >= 3 && (k % 2 != 0)))
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }

    err = fmt::format("error: invalid --blur-k (must be 0 or odd >= 3): {}", k);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_blur_fit(const cv::Mat &img, int k, std::string &err)
{
    if (img.empty()) {
        err = "error: input image is empty";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    if (k == 0)
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }

    const int min_dim = std::min(img.cols, img.rows);
    if (min_dim > 0 && k <= min_dim)
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }

    err = fmt::format(
        "error: blur kernel too large for this tool's limits: k={} image={}x{}",
        k, img.cols, img.rows);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_thresholds(int low, int high, std::string &err)
{
    if (low >= 0 && high >= 0 && low < high && high <= 255)
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    
    err = fmt::format(
        "error: invalid thresholds low: {}, high: {}. (require 0<=low<high<=255)",
        low,
        high);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_contours_thresh_mode(std::string_view mode, std::string &err)
{
    if (mode == "otsu" || mode == "adaptive" || mode == "manual")
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    
    err = fmt::format("error: invalid --thresh (must be otsu|adaptive|manual): {}", mode);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_draw_mode(std::string_view draw, std::string &err)
{
    if (draw == "bbox" || draw == "contour" || draw == "both")
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    
    err = fmt::format("error: invalid --draw (must be bbox|contour|both): {}", draw);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_min_area(double min_area, std::string &err)
{
    if (min_area >= 0)
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }

    err = fmt::format("error: invalid --min-area (must be >= 0): {}", min_area);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_adaptive_block(int block, std::string &err)
{
    if (block >= 3 && (block % 2 != 0))
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    
    err = fmt::format("error: invalid --block (must be odd and >= 3): {}", block);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

cvtool::core::ExitCode validate_manual_t(int t, std::string &err)
{
    if (t >= 0 && t <= 255)
    {
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    
    err = fmt::format("error: invalid --t (require 0..255): {}", t);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}

}
