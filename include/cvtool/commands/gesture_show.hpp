#pragma once 

#include "cvtool/core/exit_codes.hpp"

#include <string>

namespace cvtool::cmd {

struct GestureShowOptions
{
    int cam{0};
    std::string map_path;
    std::string size_str;
    bool mirror{false};
    std::string roi;
    bool show_debug{false};
    std::string hand_model_path;
    int stable_frames{5};
    int cooldown_ms{300};
    std::string face_model_path;
    float face_min_confidence{0.5f};
    bool enable_contextual_gestures{false};
};

}

cvtool::core::ExitCode run_gesture_show(const cvtool::cmd::GestureShowOptions &opt);