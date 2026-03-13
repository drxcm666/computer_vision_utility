#pragma once 

#include "cvtool/core/exit_codes.hpp"
#include "cvtool/core/gesture/gesture_domain.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <vector>

namespace cvtool::core::gesture
{

struct TransparentStringHash
{
    using is_transparent = void;

    std::size_t operator()(std::string_view txt) const{
        return std::hash<std::string_view>{}(txt);
    }
};

struct GestureImageBank
{
    std::filesystem::path base_dir;
    std::unordered_map<std::string, cv::Mat, TransparentStringHash, std::equal_to<>> images;
    cv::Mat fallback;
};

cvtool::core::ExitCode load_gesture_image_bank(
    const std::string &map_path,
    GestureImageBank &out_bank,
    std::vector<std::string> &warnings,
    std::string &err
);

const cv::Mat &get_gesture_image(const GestureImageBank &bank, cvtool::core::gesture::GestureID id);

}