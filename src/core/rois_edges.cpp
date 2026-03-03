#include "cvtool/core/rois_edges.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace cvtool::core::roi_edges
{

static double iou_rect(const cv::Rect &a, const cv::Rect &b)
{
    cv::Rect intersection = a & b;
    const double interA = static_cast<double>(intersection.area());
    double unionA = static_cast<double>(a.area()) + static_cast<double>(b.area()) - interA;
    double proc{0.0};

    if (unionA <= 0.0) return 0.0;

    return interA / unionA;
}

static cv::Rect pad_clamp(cv::Rect r, int pad, const cv::Rect &bounds)
{
    r.x -= pad;
    r.y -= pad;
    r.width += 2 * pad;
    r.height += 2 * pad;

    cv::Rect result = r & bounds;

    return result;
}

static std::vector<cv::Rect> merge_roi_iou(std::vector<cv::Rect> rois, double merge_iou)
{
    for (int iter = 0; iter < 4; iter++)
    {
        std::vector<cv::Rect> result;
        bool changed = false;

        for (int i = 0; i < rois.size(); i++)
        {
            if (result.empty())
            {
                result.emplace_back(rois[i]);
                continue;
            }

            bool merged{false};
            for (int j = 0; j < result.size(); j++)
            {
                if (iou_rect(result[j], rois[i]) > merge_iou)
                {
                    result[j] |= rois[i];
                    merged = true;
                    changed = true;

                    break;
                }
            }
            if (!merged)
                result.emplace_back(rois[i]);
        }
        rois = std::move(result);
        if (!changed)
            break;
    }

    return rois;
}

std::vector<cv::Rect> build_rois_edges(const cv::Mat &scene_gray, RoiEdgesParams &p)
{
    std::vector<cv::Rect> rois;
    if (scene_gray.empty()) return rois;
    
    cv::Mat blur;
    if (p.blur_k > 0){
        cv::GaussianBlur(scene_gray, blur, cv::Size(p.blur_k, p.blur_k), 0, 0);
    }
    else blur = scene_gray;

    cv::Mat edges;
    cv::Canny(blur, edges, p.low, p.high);
    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    rois.reserve(contours.size());

    cv::Size s = scene_gray.size();
    double scene_area = s.width * s.height;
    double min_area_px = p.min_area * scene_area;
    for (auto &c : contours)
    {
        cv::Rect bounding_rect = cv::boundingRect(c);

        if (static_cast<double>(bounding_rect.area()) <= min_area_px) continue;

        rois.push_back(bounding_rect);
    }

    std::sort(rois.begin(), rois.end(), [](const cv::Rect &a, const cv::Rect &b)
              { return a.area() > b.area(); });

    if (rois.size() > p.roi_max * 4)
    {
        rois.resize(p.roi_max * 4);
    }

    std::vector<cv::Rect> merged = merge_roi_iou(rois, p.merge_iou);
    std::sort(merged.begin(), merged.end(), [](const cv::Rect &a, const cv::Rect &b)
              { return a.area() > b.area(); });
    if (merged.size() > p.roi_max)
    {
        merged.resize(p.roi_max);
    }

    cv::Rect bounds{0, 0, scene_gray.cols, scene_gray.rows};
    for (auto &r : merged)
    {
        r = pad_clamp(r, p.pad, bounds);
    }

    return merged;
}

}