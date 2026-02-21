#include "cvtool/core/contours_core.hpp"
#include "cvtool/core/exit_codes.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

#include <fmt/core.h>

namespace cvtool::core::contours
{

cvtool::core::ExitCode find_contours_report(
    const cv::Mat &bin,
    double min_area,
    std::vector<ContourItem> &items,
    ContourStats &stats,
    std::string &err)
{
    items.clear();
    stats = cvtool::core::contours::ContourStats{};
    err.clear();

    if (bin.empty())
    {
        err = "error: input image is empty";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    if (bin.channels() != 1)
    {
        err = fmt::format("error: input image must be single-channel: {}", bin.channels());
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    if (bin.depth() != CV_8U)
    {
        err = fmt::format("error: input image must be CV_8UC1 (got type={} channels={} depth={})",
                  bin.type(), bin.channels(), bin.depth());
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    try
    {
        cv::Mat tmp = bin.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(tmp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        stats.contours_total = static_cast<int>(contours.size());

        int kept_id{0};
        for (const auto &i : contours)
        {
            double area = cv::contourArea(i);
            if (area >= min_area)
            {
                cv::Rect bbox = cv::boundingRect(i);
                items.push_back({kept_id++, area, bbox, i});
            }
        }
        
        stats.contours_kept = static_cast<int>(items.size());

        if (items.empty())
        {
            stats.area_min = 0.0;
            stats.area_mean = 0.0;
            stats.area_max = 0.0;       
        }
        else
        {
            double min_val = items[0].area;
            double max_val = items[0].area;
            double sum_val{0.0};

            for (const auto &i : items)
            {
                if (min_val > i.area)
                    min_val = i.area;

                if (max_val < i.area)
                    max_val = i.area;

                sum_val += i.area;
            }
            
            stats.area_min = min_val;
            stats.area_max = max_val;
            stats.area_mean = sum_val / static_cast<double>(items.size());
        }
    }
    catch(const cv::Exception &e)
    {
        err = fmt::format("OpenCV error in findContours: {}", e.what());
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    
    return cvtool::core::ExitCode::Ok;
}

}