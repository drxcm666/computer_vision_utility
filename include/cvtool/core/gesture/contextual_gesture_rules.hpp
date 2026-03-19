#pragma once

#include "cvtool/core/gesture/gesture_domain.hpp"
#include "cvtool/core/gesture/face_landmarks.hpp"
#include "cvtool/core/gesture/hand_landmarks.hpp"

namespace cvtool::core::gesture
{

GestureID classify_contextual_gesture(
    const HandLandmarkResult &hand_data,
    const FaceLandmarkResult &face_data);


}