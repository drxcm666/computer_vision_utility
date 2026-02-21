#include "cvtool/core/threshold.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/exit_codes.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <fmt/format.h>


cvtool::core::ExitCode make_binary_mask(
    const cv::Mat &src, 
    const std::string &mode, 
    int blur_k, bool invert, int block, double c, int t, 
    cv::Mat &out_bin, std::string &err
)
{
    const auto channels_code = cvtool::core::validate::validate_gray_channels(src.channels(), err);
    if (channels_code != cvtool::core::ExitCode::Ok)
    {
        return channels_code;
    }

    const auto blur_code = cvtool::core::validate::validate_blur_k(blur_k, err);
    if (blur_code != cvtool::core::ExitCode::Ok)
    {
        err = fmt::format("error: threshold parameter --blur-k: {}", err);
        return blur_code;
    }

    if (mode == "manual")
    {
        const auto t_code = cvtool::core::validate::validate_manual_t(t, err);
        if (t_code != cvtool::core::ExitCode::Ok)
        {
            err = fmt::format("error: threshold parameter --t: {}", err);
            return t_code;
        }
    }
    else if (mode == "adaptive")
    {
        const auto b_code = cvtool::core::validate::validate_adaptive_block(block, err);
        if (b_code != cvtool::core::ExitCode::Ok)
        {
            err = fmt::format("error: threshold parameter --block: {}", err);
            return b_code;
        }
    }

    try
    {
        cv::Mat gray;
        if (src.channels() == 1){
            gray = src.clone();
        } else if (src.channels() == 3){
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        } else if (src.channels() == 4){
            cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
        }
        if (gray.empty() || gray.channels() != 1){
            err = "error: grayscale conversion failed";
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }

        if (gray.depth() != CV_8U)
        {
            cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX);
            gray.convertTo(gray, CV_8U);
        }

        cv::Mat blur_used;
        if (blur_k > 0)
            cv::GaussianBlur(gray, blur_used, cv::Size(blur_k, blur_k), 0);
        else
            blur_used = gray;

        cv::Mat bin;

        if (mode == "otsu")
        {
            cv::threshold(blur_used, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        }
        else if (mode == "manual")
        {
            cv::threshold(blur_used, bin, t, 255, cv::THRESH_BINARY);
        }
        else if (mode == "adaptive")
        {
            cv::adaptiveThreshold(blur_used, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block, c);
        }
        else
        {
            err = fmt::format("error: invalid --thresh (must be otsu|adaptive|manual): {}", mode);
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
        
        if (invert)
            cv::bitwise_not(bin, bin);
        
        if (!bin.empty() && bin.channels() == 1 && bin.size()==blur_used.size())
        {
            out_bin = bin.clone();
            err.clear();
            return cvtool::core::ExitCode::Ok;
        }

        err = "error: threshold failed (empty or invalid mask)";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    catch(const cv::Exception &e)
    {
        err = std::string("error: threshold failed (") + e.what() + ")";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
}