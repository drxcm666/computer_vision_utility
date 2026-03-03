#include "cvtool/core/image_convert.hpp"

#include <opencv2/imgproc.hpp>

#include <fmt/format.h>

namespace cvtool::core::img
{

cv::Mat to_gray(const cv::Mat &img)
{
    cv::Mat gray;
    if (img.channels() == 1)
    {
        return img;
    }
    else if (img.channels() == 3)
    {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else if (img.channels() == 4)
    {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        return cv::Mat{};
    }

    return gray;
}


bool to_bgr(const cv::Mat &img, cv::Mat &dst, std::string &err)
{
    switch (img.channels())
    {
    case 1:
        cv::cvtColor(img, dst, cv::COLOR_GRAY2BGR);
        return true;
    case 3:
        dst = img.clone();
        return true;
    case 4:
        cv::cvtColor(img, dst, cv::COLOR_BGRA2BGR);
        return true;
    default:
        err = fmt::format("error: unsupported channels: {}", img.channels());
        return false;
    }
}

}