#include "cvtool/core/edges_pipeline.hpp"
#include "cvtool/core/validate.hpp"

#include <opencv2/imgproc.hpp>

#include <string>

namespace cvtool::core
{

ExitCode edges_frame_to_gray(
    const cv::Mat &frame,
    int low,
    int high,
    int blur_k,
    cv::Mat &out_gray,
    std::string &err
)
{
    if(frame.empty())
    {
        err = "error: input frame is empty";
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }

    auto v = cvtool::core::validate::validate_thresholds(low, high, err);
    if (v != cvtool::core::ExitCode::Ok)
        return v;

    v = cvtool::core::validate::validate_blur_k(blur_k, err);
    if (v != cvtool::core::ExitCode::Ok)
        return v;

    auto channels = frame.channels();
    v = cvtool::core::validate::validate_gray_channels(channels, err);
    if (v != cvtool::core::ExitCode::Ok)
        return v;

    try
    {
        cv::Mat gray;
        if (channels == 1){
            gray = frame.clone();
        } else if (channels == 3){
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else if (channels == 4){
            cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
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

        cv::Mat edges;
        cv::Canny(blur_used, edges, low, high);

        if (edges.empty()
            || edges.channels() != 1
            || edges.depth() != CV_8U
            || edges.size() != blur_used.size())
        {
            err = "error: edges pipeline failed";
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }

        out_gray = edges;
        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    catch(const cv::Exception &e)
    {
        err = std::string("error: edges pipeline failed (") + e.what() + ")";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
}

ExitCode edges_frame_to_bgr(
    const cv::Mat &frame,
    int low, 
    int high, 
    int blur_k, 
    cv::Mat &out_bgr,
    std::string &err 
)
{
    cv::Mat edges_gray;
    const cvtool::core::ExitCode rc = edges_frame_to_gray(frame, low, high, blur_k, edges_gray, err);
    if (rc != cvtool::core::ExitCode::Ok)
        return rc;

    try
    {
        cv::cvtColor(edges_gray, out_bgr, cv::COLOR_GRAY2BGR);

        if (out_bgr.channels() != 3
            || out_bgr.depth() != CV_8U
            || out_bgr.size() != frame.size())
        {
            err = "error: edges pipeline failed";
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }

        err.clear();
        return cvtool::core::ExitCode::Ok;
    }
    catch(const cv::Exception &e)
    {
        err = std::string("error: edges pipeline failed (") + e.what() + ")";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
}

}