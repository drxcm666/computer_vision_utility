#include "cvtool/core/gesture/gesture_bank.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <fmt/format.h>
#include <fmt/std.h>

#include <unordered_set>
#include <string_view>
#include <fstream>

namespace cvtool::core::gesture
{


const cv::Mat &get_gesture_image(const GestureImageBank &bank, cvtool::core::gesture::GestureID id)
{
    std::string_view gest_name = to_asset_key(id);

    auto it = bank.images.find(gest_name);
    if (it != bank.images.end() && !it->second.empty()){
        return it->second;
    } 
    
    return bank.fallback;
}

cvtool::core::ExitCode load_gesture_image_bank(
    const std::string &map_path,
    GestureImageBank &out_bank,
    std::vector<std::string> &warnings,
    std::string &err
)
{
    warnings.clear();
    out_bank.fallback.release();
    out_bank.images.clear();

    const std::unordered_set<std::string> whitelist{
        "open_palm", "fist", "peace", "thumbs_up", "monkey", "none", "unknown"};

    std::ifstream file(map_path);
    if (!file)
    {
        err = "error: cannot open the instruction file";
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }
    
    nlohmann::json j;
    try
    {
        j = nlohmann::json::parse(file);
    }
    catch(const std::exception& e)
    {
        err = e.what();
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    if (!j.is_object())
    {
        err = "error: JSON file does not have a dictionary";
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }
    
    std::filesystem::path mp(map_path);
    out_bank.base_dir = mp.parent_path();

    for (auto &[gesture, path] : j.items())
    {
        if (!path.is_string())
        {
            warnings.push_back(fmt::format(
                "[gesture_bank] warn: the gesture's path in the file is not a string; gest={}; path={}",
                gesture, path.dump()));
            continue;
        }
        if (!whitelist.contains(gesture))
        {
            warnings.push_back(fmt::format(
                "[gesture_bank] warn: unknown gesture ignored; gest={}; path={}", 
                gesture, path.dump()));
            continue;
        }

        std::filesystem::path p = path.get<std::string>();
        std::filesystem::path image_path;
        if (p.is_absolute())
        {
            image_path = p;
        }
        else image_path = out_bank.base_dir / p;
        image_path = image_path.lexically_normal();

        std::error_code ec;
        const bool exists = std::filesystem::exists(image_path, ec);
        if (ec)
        {
            warnings.push_back(fmt::format(
                "[gesture_bank] warn: cannot access input path; gest={}; path={} ({})", 
                gesture, image_path.string(), ec.message()));
            continue;
        }
        if (!exists)
        {
            warnings.push_back(fmt::format(
                "[gesture_bank] warn: input file not found; gest={}; path={}", 
                gesture, image_path.string()));
            continue;
        }

        cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            warnings.push_back(fmt::format(
                "[gesture_bank] warn: cannot read image; gest={}; path={}", 
                gesture, image_path.string()));
            continue;
        }

        cv::Mat bgr_img;
        if (img.channels() == 1){
            cv::cvtColor(img, bgr_img, cv::COLOR_GRAY2BGR);
        } else if (img.channels() == 4){
            cv::cvtColor(img, bgr_img, cv::COLOR_BGRA2BGR);
        } else {
            bgr_img = img;
        }

        if (gesture == "none")
            out_bank.fallback = bgr_img;
        else
            out_bank.images[gesture] = bgr_img;
    }

    if (out_bank.fallback.empty())
    {
        warnings.push_back(fmt::format(
            "[gesture_bank] warn: no 'none' image configured, will use default fallback"));
    }
    if (out_bank.fallback.empty())
    {
        warnings.push_back(fmt::format(
            "[gesture_bank] warn: no gesture images loaded at all"));
    }
    
    return cvtool::core::ExitCode::Ok;
}
}