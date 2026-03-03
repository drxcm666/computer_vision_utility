#include "cvtool/core/match/match_render.hpp"
#include "cvtool/core/image_convert.hpp"

#include <fmt/format.h>


namespace cvtool::core::match_render 
{

cvtool::core::ExitCode render (
    const std::vector<cv::Rect> &rois,
    const std::vector<cvtool::core::templ_match::MatchBest> &hits_topk,
    const cv::Mat &scene,
    bool draw_roi,
    std::string_view draw_mode,
    int thickness,
    double font_scale,
    cv::Mat &vis,
    std::string &err
)
{
    if (!cvtool::core::img::to_bgr(scene, vis, err))
    {
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    if (draw_roi)
    {
        for (int i = 0; i < static_cast<int>(rois.size()); i++)
        {
            const auto &h = rois[i];
            cv::rectangle(vis, h, cv::Scalar(0, 0, 255), 1);
        }
    }
    for (int i = 0; i < static_cast<int>(hits_topk.size()); i++)
    {
        const auto &h = hits_topk[i];

        cv::rectangle(vis, h.bbox, cv::Scalar(0, 255, 0), thickness);

        int y = (h.bbox.y > 5) ? (h.bbox.y - 5) : 0;
        cv::Point text_pos(h.bbox.x, y);

        if (draw_mode != "bbox")
        {
            const bool with_score = (draw_mode == "bbox+label+score");
            const auto text = with_score ? fmt::format("#{} conf={:.2f} s={:.2f}", i, h.confidence, h.scale)
                                         : fmt::format("#{} s={:.2f}", i, h.scale);

            cv::putText(
                vis,
                text,
                text_pos,
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                cv::Scalar(0, 255, 0),
                1);
        }
    }

    return cvtool::core::ExitCode::Ok;
}

}