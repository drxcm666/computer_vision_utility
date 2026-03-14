#pragma once 

#include "cvtool/core/gesture/gesture_domain.hpp"
#include "cvtool/core/gesture/hand_landmarks.hpp"

namespace cvtool::core::gesture
{

struct FingerState
{
    bool thumb_extended{false};
    bool index_extended{false};
    bool middle_extended{false};
    bool ring_extended{false};
    bool pinky_extended{false};
};

FingerState extract_finger_state(const cvtool::core::gesture::HandLandmarkResult &data);

cvtool::core::gesture::GestureID classify_hand_gesture(const cvtool::core::gesture::HandLandmarkResult &data);



}