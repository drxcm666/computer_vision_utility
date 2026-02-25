#include "cvtool/core/template_match.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace cvtool::core::templ_match 
{

static double iou_rect(const cv::Rect &a, const cv::Rect &b)
{
    const cv::Rect inter = a & b;
    if (inter.empty()) return 0.0;
    const double ia = static_cast<double>(inter.area());
    const double ua = static_cast<double>(a.area()) + static_cast<double>(b.area()) - ia;
    return (ua > 0.0) ? (ia / ua) : 0.0;
}

static double confidence_from_raw(int method, const double raw)
{
    double conf{0.0};
    if (method == cv::TM_SQDIFF_NORMED || method == cv::TM_SQDIFF) conf = 1 - raw;
    else if (method == cv::TM_CCORR_NORMED) conf = raw;
    else conf = (raw + 1) / 2;

    return std::clamp(conf, 0.0, 1.0);
}

MatchBest match_best(const cv::Mat &scene, const cv::Mat &templ, int method)
{
    cv::Mat result;
    cv::matchTemplate(scene, templ, result, method);

    double minV{}, maxV{}; 
    cv::Point minP{}, maxP{};
    cv::minMaxLoc(result, &minV, &maxV, &minP, &maxP);

    const bool is_sqdiff = (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED);
    const double raw = is_sqdiff ? minV : maxV;
    const cv::Point loc = is_sqdiff ? minP : maxP;
    cv::Rect bbox{loc.x, loc.y, templ.cols, templ.rows};
    const double conf = confidence_from_raw(method, raw);

    return{bbox, raw, conf};
}

std::vector<MatchBest> match_topk(
    const cv::Mat &scene,
    const cv::Mat &templ,
    int method, 
    int max_results,
    double min_score,
    cv::Mat *out_result 
)
{
    cv::Mat result;
    cv::matchTemplate(scene, templ, result, method);

    if (out_result)
    {
        *out_result = result;
    }
    

    cv::Mat work = result.clone();
    std::vector<MatchBest> hits;

    const bool is_sqdiff = (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED);
    const double worst = is_sqdiff ? 1.0 : -1.0;

    for (int i = 0; i < max_results; i++)
    {
        double minV{}, maxV{}; 
        cv::Point minP{}, maxP{};
        cv::minMaxLoc(work, &minV, &maxV, &minP, &maxP);

        const double raw = is_sqdiff ? minV : maxV;
        const cv::Point loc = is_sqdiff ? minP : maxP;
        

        const double conf = confidence_from_raw(method, raw);
        if (conf < min_score) break;
    
        cv::Rect bbox{loc.x, loc.y, templ.cols, templ.rows};
        hits.push_back(MatchBest{bbox, raw, conf});

        const int rx = std::max(1, templ.cols / 4);
        const int ry = std::max(1, templ.rows / 4);
        cv::Rect sup{loc.x - rx, loc.y - ry, 2 * rx + 1, 2 * ry + 1};

        sup &= cv::Rect{0, 0, work.cols, work.rows};
        work(sup).setTo(worst);
    }
    
    return hits;
}

std::vector<MatchBest> nms_iou(
    const std::vector<MatchBest> &hits,
    double iou_thr,
    int max_keep
)
{
    std::vector<int> idx(hits.size());
    for (int i = 0; i < static_cast<int>(idx.size()); i++){
        idx[i] = i;
    }

    std::sort(idx.begin(), idx.end(), [&](int i, int j){ return hits[i].confidence > hits[j].confidence; });

    std::vector<MatchBest> out;
    out.reserve(std::min(max_keep, static_cast<int>(hits.size())));
    

    for (int id : idx)
    {
        bool ok = true;
        for (const auto &kept : out)
        {
            if (iou_rect(hits[id].bbox, kept.bbox) >= iou_thr)
            {
                ok = false;
                break;
            }
        }
        if (ok)
        {
            out.push_back(hits[id]);
            if(static_cast<int>(out.size()) == max_keep) break; 
        }
    }

    return out;
}

}