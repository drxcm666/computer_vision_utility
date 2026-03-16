#include "cvtool/core/gesture/gesture_stabilizer.hpp"

namespace cvtool::core::gesture
{


void GestureStabilizer::reset()
{
    stable_gesture_ = GestureID::None;
    candidate_gesture_ = GestureID::None;
    candidate_count_ = 0;
    last_change_time_ = std::chrono::steady_clock::time_point();
}


StabilizerResult GestureStabilizer::update(
    GestureID gesture, std::chrono::steady_clock::time_point time_now)
{
    if (gesture == stable_gesture_)
    {
        candidate_gesture_ = GestureID::None;
        candidate_count_ = 0;
    }
    else if (gesture == candidate_gesture_)
    {
        if (candidate_count_ < stable_frames_required_)
            candidate_count_++;
    }
    else
    {
        candidate_gesture_ = gesture;
        candidate_count_ = 1;
    }

    if ((candidate_count_ >= stable_frames_required_) && 
        ((time_now - last_change_time_) >= cooldown_))
    {
        stable_gesture_ = candidate_gesture_;
        last_change_time_ = time_now;
        candidate_count_ = 0;
        candidate_gesture_ = GestureID::None;
    }
    
    
    return {stable_gesture_, candidate_gesture_, candidate_count_};
}


}