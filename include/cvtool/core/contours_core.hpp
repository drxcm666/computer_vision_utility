#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

#include <string>
#include <vector>

namespace cvtool::core::contours
{

struct ContourItem
{
    int id{0};
    double area{0.0};
    cv::Rect bbox;
    std::vector<cv::Point> contour;
};

struct ContourStats
{
    int contours_total{0};
    int contours_kept{0};
    double area_min{0.0};
    double area_mean{0.0};
    double area_max{0.0};
};

cvtool::core::ExitCode find_contours_report(
    const cv::Mat &bin, 
    double min_area, 
    std::vector<ContourItem> &items, 
    ContourStats &stats, 
    std::string &err
);

}
