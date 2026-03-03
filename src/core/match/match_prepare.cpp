#include "cvtool/core/match/match_prepare.hpp"
#include "cvtool/core/image_convert.hpp"

#include <fmt/core.h>

namespace cvtool::core::match_preparate
{

cvtool::core::ExitCode preparate_for_match(
    const cv::Mat &img, 
    std::string_view mode, 
    cv::Mat &out, 
    std::string &err
)
{
    if (mode == "gray")
    {
        out = cvtool::core::img::to_gray(img);
        if (!out.empty())
            return cvtool::core::ExitCode::Ok;
        err = fmt::format("can't convert to gray (channels={})", img.channels());
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    bool bgr_done = cvtool::core::img::to_bgr(img, out, err);
    if (!bgr_done)
    {
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    
    return cvtool::core::ExitCode::Ok;
}

}