#include "cvtool/core/match/match_heatmap.hpp"
#include "cvtool/core/image_io.hpp"

#include <opencv2/imgproc.hpp>

namespace cvtool::core::match_heatmap
{

cvtool::core::ExitCode write_heatmap(
    const cv::Mat &result,
    int method,
    const std::string &out_path,
    std::string &err)
{
    if (result.empty())
    {
        err = "error: heatmap requested but result matrix is empty";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    cv::Mat heat = result.clone();
    if (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED)
    {
        heat = 1.0 - heat;
    }

    cv::Mat heat_u8;
    cv::normalize(heat, heat_u8, 0, 255, cv::NORM_MINMAX);
    heat_u8.convertTo(heat_u8, CV_8U);

    cv::Mat heat_color;
    cv::applyColorMap(heat_u8, heat_color, cv::COLORMAP_JET);

    return cvtool::core::image_io::write_image(out_path, heat_color, err);
}

}