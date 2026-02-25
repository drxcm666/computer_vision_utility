#include "cvtool/core/validate.hpp"

#include <opencv2/imgproc.hpp>

#include <fmt/format.h>
#include <algorithm>
#include <charconv>

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

cvtool::core::ExitCode validate_01(std::string_view name, double v, std::string &err)
{
    if (v < 0.0 || v > 1.0)
    {
        err = fmt::format("error: {} out of range [0...1]: {}", name, v);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    return cvtool::core::ExitCode::Ok;
}

cvtool::core::ExitCode  validate_max_results(int n, std::string &err)
{
    if (n < 1)
    {
        err = fmt::format("error: max-result must be >= 1: {}", n);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    return cvtool::core::ExitCode::Ok;
}

cvtool::core::ExitCode validate_mode_match(std::string_view mode, std::string &err)
{
    if (mode != "gray" && mode != "color")
    {
        err = fmt::format("error: invalid mode: {} (must be gray|color)", mode);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    return cvtool::core::ExitCode::Ok;
}

cvtool::core::ExitCode validate_method_match(std::string_view method_str, int &method_out, std::string &err)
{
    if (method_str == "ccoeff_normed")
    {
        method_out = cv::TM_CCOEFF_NORMED;
        return cvtool::core::ExitCode::Ok;
    }
    if (method_str == "ccorr_normed")
    {
        method_out = cv::TM_CCORR_NORMED;
        return cvtool::core::ExitCode::Ok;
    }
    if (method_str == "sqdiff_normed")
    {
        method_out = cv::TM_SQDIFF_NORMED;
        return cvtool::core::ExitCode::Ok;
    }

    err = fmt::format("error: invalid method: {} (must be ccoeff_normed|ccorr_normed|sqdiff_normed)", method_str);
    return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
}


static std::string_view trim_view(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.remove_suffix(1);
    return s;
}

cvtool::core::ExitCode validate_roi(std::string_view str, cv::Rect &out, std::string &err)
{
    std::array<std::string_view, 4> result;
    std::size_t pos{0};
    std::size_t found{0};
    int i{0};

    while ((found = str.find(',', pos)) != std::string_view::npos)
    {
        if (i >= 4) {
            err = "error: too many values for ROI (expected 4: x,y,w,h)";
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }

        result[i++] = str.substr(pos, found - pos);
        pos = found + 1;
    }
    result[i] = str.substr(pos);

    if (i != 3) 
    { 
        err = "error: roi must be 4 integers: x,y,w,h"; 
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    std::array<int, 4> numbers;
    for (int i = 0; i < numbers.size(); i++)
    {
        auto tok = trim_view(result[i]);
        if (tok.empty()) 
        { 
            err = "error: roi has empty value (expected x,y,w,h)"; 
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }

        const char *b = tok.data();
        const char *e = tok.data() + tok.size();
        auto [ptr, ec] = std::from_chars(b, e, numbers[i]);
        
        if (ec != std::errc() || ptr != e)
        {
            err = "error: ROI coordinates must be valid numbers";
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
        
    }

    if (numbers[2] <= 0 || numbers[3] <= 0)
    {
        err = "error: roi width/height must be >0";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    
    out = cv::Rect{numbers[0], numbers[1], numbers[2], numbers[3]};

    return cvtool::core::ExitCode::Ok;
}

cvtool::core::ExitCode validate_draw_match(std::string_view draw, std::string &err)
{
    if (draw != "bbox" && draw != "bbox+label" && draw != "bbox+label+score")
    {
        err = fmt::format("error: invalid draw: {} (use bbox|bbox+label|bbox+label+score)", draw);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    return cvtool::core::ExitCode::Ok;
}

cvtool::core::ExitCode validate_thickness(int thickness, std::string &err)
{
    if (thickness < 1)
    {
        err = fmt::format("error: thickness must be >= 1: {}", thickness);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    return cvtool::core::ExitCode::Ok;
}

cvtool::core::ExitCode validate_font_scale(double fs, std::string &err)
{
    if (fs <= 0.0)
    {
        err = fmt::format("error: font-scale must be > 0: {}", fs);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    return cvtool::core::ExitCode::Ok;
}

}