#pragma once

#include "cvtool/core/gesture/gesture_domain.hpp"

#include <chrono>

namespace cvtool::core::gesture
{

struct StabilizerResult
{
    GestureID stable_gesture{GestureID::None};
    GestureID candidate_gesture{GestureID::None};
    int candidate_count{0};
};


class GestureStabilizer
{
private:
    GestureID stable_gesture_{GestureID::None};
    GestureID candidate_gesture_{GestureID::None};
    int candidate_count_{0};
    std::chrono::steady_clock::time_point last_change_time_{};
    int stable_frames_required_{};
    std::chrono::milliseconds cooldown_{};


public:
    GestureStabilizer(int stable_frames, int cooldown_ms) 
        : stable_frames_required_(stable_frames),
          cooldown_(cooldown_ms)
        {
            if (stable_frames_required_ < 1) stable_frames_required_ = 1;
            if (cooldown_ < std::chrono::milliseconds::zero()) cooldown_ = std::chrono::milliseconds::zero();
            reset();
        }

    StabilizerResult update(
        GestureID gesture, std::chrono::steady_clock::time_point time_now);

    void reset();


};


}