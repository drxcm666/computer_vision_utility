#include "cvtool/core/match/match_json.hpp"
#include "cvtool/core/validate.hpp"

#include <nlohmann/json.hpp>
#include <iomanip>
#include <fstream>

namespace cvtool::core::match_json
{

cvtool::core::ExitCode write_match_json(
    const cvtool::cmd::MatchOptions &opt,
    const cv::Size &scene_size,
    const cv::Size &templ_size,
    const std::vector<cvtool::core::templ_match::MatchBest> &hits,
    const std::vector<cv::Rect> &rois,
    bool roi_fallback_used,
    const std::string &roi_source,
    std::string &err)
{
    nlohmann::ordered_json j;

    j["command"] = "match";
    j["input"] = opt.in_path;
    j["template"] = opt.templ_path;
    j["output"] = opt.out_path;
    j["params"] = {{"mode", opt.mode}, {"method", opt.method}, {"max_results", opt.max_results}, {"min_score", opt.min_score}, {"nms", opt.nms}, {"draw", opt.draw}, {"thickness", opt.thickness}, {"font_scale", opt.font_scale}};
    j["template_size"] = {{"w", templ_size.width}, {"h", templ_size.height}};
    j["scene_size"] = {{"w", scene_size.width}, {"h", scene_size.height}};

    if (!opt.roi.empty())
    {
        cv::Rect r;
        std::string err_dummy;
        if (cvtool::core::validate::validate_roi(opt.roi, r, err_dummy) == cvtool::core::ExitCode::Ok)
        {
            j["params"]["roi"] = {{"x", r.x}, {"y", r.y}, {"w", r.width}, {"h", r.height}};
        }
        else
        {
            j["params"]["roi"] = nlohmann::ordered_json::object();
        }
    }
    else
    {
        j["params"]["roi"] = nlohmann::ordered_json::object();
    }

    j["roi_auto"] = opt.roi_auto;
    j["roi_fallback"] = roi_fallback_used;
    j["rois"] = nlohmann::json::array();
    for (auto &r : rois)
    {
        j["rois"].push_back({
            {"x", r.x}, {"y", r.y}, {"w", r.width}, {"h", r.height}, {"source", roi_source}
        });
    }

    j["matches"] = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(hits.size()); ++i)
    {
        const auto &h = hits[i];
        j["matches"].push_back({{"id", i},
                                {"bbox", {{"x", h.bbox.x}, {"y", h.bbox.y}, {"w", h.bbox.width}, {"h", h.bbox.height}}},
                                {"raw_score", h.raw_score},
                                {"confidence", h.confidence},
                                {"scale", h.scale},
                                {"template_size", {{"w", h.template_size.width}, {"h", h.template_size.height}}}});
    }
    j["stats"] = {{"found", (int)hits.size()}};

    std::ofstream file(opt.json_path);
    if (!file)
    {
        err = "error: cannot open json output '" + opt.json_path + "'";
        return cvtool::core::ExitCode::CannotWriteOutput;
    }

    file << std::setw(4) << j << '\n';
    if (!file.good())
    {
        err = "error: failed to write json output: " + opt.json_path;
        return cvtool::core::ExitCode::CannotWriteOutput;
    }

    return cvtool::core::ExitCode::Ok;
}

}