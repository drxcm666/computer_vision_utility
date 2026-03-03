#include "cvtool/commands/match_validate.hpp"
#include "cvtool/core/validate.hpp"

#include <fmt/format.h>

namespace cvtool::cmd::match_validate
{

cvtool::core::ExitCode validate_match_options(
    const cvtool::cmd::MatchOptions &opt,
    int &method,
    double &scale_min, double &scale_max, double &scale_step,
    int &count, int &per_scale_top,
    std::string &err

)
{
    auto mins_code = cvtool::core::validate::validate_01("min-score", opt.min_score, err);
    if (mins_code != cvtool::core::ExitCode::Ok)
    {
        return mins_code;
    }
    auto nms_code = cvtool::core::validate::validate_01("nms", opt.nms, err);
    if (nms_code != cvtool::core::ExitCode::Ok)
    {
        return nms_code;
    }
    auto maxres_code = cvtool::core::validate::validate_max_results(opt.max_results, err);
    if (maxres_code != cvtool::core::ExitCode::Ok)
    {
        return maxres_code;
    }
    auto mode_code = cvtool::core::validate::validate_mode_match(opt.mode, err);
    if (mode_code != cvtool::core::ExitCode::Ok)
    {
        return mode_code;
    }
    auto method_code = cvtool::core::validate::validate_method_match(opt.method, method, err);
    if (method_code != cvtool::core::ExitCode::Ok)
    {
        return method_code;
    }
    auto draw_code = cvtool::core::validate::validate_draw_match(opt.draw, err);
    if (draw_code != cvtool::core::ExitCode::Ok)
    {
        return draw_code;
    }
    auto thickness_code = cvtool::core::validate::validate_thickness(opt.thickness, err);
    if (thickness_code != cvtool::core::ExitCode::Ok)
    {
        return thickness_code;
    }
    auto fs_code = cvtool::core::validate::validate_font_scale(opt.font_scale, err);
    if (fs_code != cvtool::core::ExitCode::Ok)
    {
        return fs_code;
    }
    auto roi_merge_code = cvtool::core::validate::validate_01("roi-merge-iou", opt.roi_merge_iou, err);
    if (roi_merge_code != cvtool::core::ExitCode::Ok)
    {
        return roi_merge_code;
    }
    auto minarea_code = cvtool::core::validate::validate_01("roi-min-area", opt.roi_min_area, err);
    if (minarea_code != cvtool::core::ExitCode::Ok)
    {
        return minarea_code;
    }
    auto max_code = cvtool::core::validate::validate_max_results(opt.roi_max, err);
    if (max_code != cvtool::core::ExitCode::Ok)
    {
        return max_code;
    }
    auto pad_code = cvtool::core::validate::validate_nonneg("roi-pad", opt.roi_pad, err);
    if (pad_code != cvtool::core::ExitCode::Ok)
    {
        return pad_code;
    }
    auto blur_code = cvtool::core::validate::validate_blur_k(opt.roi_edges_blur_k, err);
    if (blur_code != cvtool::core::ExitCode::Ok)
    {
        return blur_code;
    }
    auto thr_code = cvtool::core::validate::validate_thresholds(opt.roi_edges_low, opt.roi_edges_high, err);
    if (thr_code != cvtool::core::ExitCode::Ok)
    {
        return thr_code;
    }

    auto scale_code = cvtool::core::validate::validate_scale_range(opt.scales, scale_min, scale_max, scale_step, err);
    if (scale_code != cvtool::core::ExitCode::Ok)
    {
        return scale_code;
    }

    count = 1 + (static_cast<int>(std::floor((scale_max - scale_min) / scale_step + 1e-9)));
    if (opt.max_scales > 0 && count > opt.max_scales)
    {
        err = fmt::format("error: too many scales (count={}, max={})", count, opt.max_scales);
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    per_scale_top = (opt.per_scale_top > 0) ? opt.per_scale_top : opt.max_results * 5;


    return cvtool::core::ExitCode::Ok;
}

    
}