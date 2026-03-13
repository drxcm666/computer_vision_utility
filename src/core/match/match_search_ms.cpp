#include "cvtool/core/match/match_search_ms.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

namespace cvtool::core::match_search_ms
{

cvtool::core::ExitCode search_multiscale(
    const cv::Mat &scene_proc,
    const cv::Mat &templ_proc,
    int method,
    const std::vector<cv::Rect> &search_rois,
    double scale_min,
    double scale_step,
    int count,
    int per_scale_top,
    double min_score,
    bool need_heatmap,
    std::vector<cvtool::core::templ_match::MatchBest> &out_all,
    cv::Mat &out_best_result,
    int &out_valid_scales,
    std::string &err
)
{
    double best_conf{-1.0};
    cv::Mat result_buffer;
    for (auto &r : search_rois)
    {
        const cv::Mat sub_scene = scene_proc(r);

        for (int i = 0; i < count; i++)
        {
            const double scale = scale_min + i * scale_step;
            const int w_s = static_cast<int>(std::lround(templ_proc.cols * scale));
            const int h_s = static_cast<int>(std::lround(templ_proc.rows * scale));

            if (w_s < 2 || h_s < 2)
                continue;
            if (w_s > sub_scene.cols || h_s > sub_scene.rows)
                continue;

            out_valid_scales++;

            cv::Mat templ_s;
            if (std::abs(scale - 1) < 1e-5)
            {
                templ_s = templ_proc;
            }
            else
            {
                const auto interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;
                cv::resize(templ_proc, templ_s, cv::Size(w_s, h_s), 0, 0, interp);
            }
            
            cv::Mat *out_res_s = need_heatmap ? &result_buffer : nullptr;
            auto cands = cvtool::core::templ_match::match_topk(sub_scene, templ_s, method, per_scale_top, min_score, out_res_s);

            for (auto &hh : cands)
            {
                hh.scale = scale;
                hh.template_size = templ_s.size();
                hh.bbox.x += r.x;
                hh.bbox.y += r.y;
            }

            out_all.insert(out_all.end(), cands.begin(), cands.end());

            if (need_heatmap)
            {
                bool new_record = false;

                if (out_best_result.empty() && !result_buffer.empty())
                    new_record = true;

                if (!cands.empty() && cands[0].confidence > best_conf)
                    new_record = true;

                if (new_record)
                {
                    if (!cands.empty()) best_conf = cands[0].confidence;
                    out_best_result = result_buffer.clone();
                }
            }
        }
    }

    if (out_valid_scales == 0)
    {
        err = "error: no valid scales (template never fits into scene/roi)";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    return cvtool::core::ExitCode::Ok;
}
}