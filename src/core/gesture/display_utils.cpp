#include "cvtool/core/gesture/display_utils.hpp"

#include <algorithm>
#include <cmath>

namespace cvtool::core::gesture
{

cv::Mat letterbox(const cv::Mat &source, int targetWidth, int targetHeight)
{
    if (source.empty())
    {
        return cv::Mat::zeros(targetHeight, targetWidth, CV_8UC3);
    }

    double scale = std::min(
        (static_cast<double>(targetWidth) / source.cols), (static_cast<double>(targetHeight) / source.rows));

    int newWidth = std::round(source.cols * scale);
    int newHeight = std::round(source.rows * scale);

    cv::Mat resized;
    int interpolation = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(source, resized, cv::Size(newWidth, newHeight), 0, 0, interpolation);

    int padTop = (targetHeight - newHeight) / 2;
    int padBottom = targetHeight - newHeight - padTop;
    int padLeft = (targetWidth - newWidth) / 2;
    int padRight = targetWidth - newWidth - padLeft;

    cv::Mat result;
    cv::copyMakeBorder(resized, result, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return result;
}

}