#pragma once 

#include "cvtool/core/gesture/gesture_domain.hpp"
#include "cvtool/core/gesture/hand_landmarks.hpp"

namespace cvtool::core::gesture
{

enum HandLandmarkIndex
{
    Wrist,
    ThumbCmc, ThumbMcp, ThumbIp, ThumbTip,
    IndexMcp, IndexPip, IndexDip, IndexTip,
    MiddleMcp, MiddlePip, MiddleDip, MiddleTip,
    RingMcp, RingPip, RingDip, RingTip,
    PinkyMcp, PinkyPip, PinkyDip, PinkyTip
};

struct FingerState
{
    bool thumb_extended{false};
    bool index_extended{false};
    bool middle_extended{false};
    bool ring_extended{false};
    bool pinky_extended{false};
};


struct ClassifierResult
{
    GestureID gesture;
    FingerState state;

};

float distance(const cv::Point2f &point1, const cv::Point2f &point2);
float compute_palm_scale(const cvtool::core::gesture::HandLandmarkResult &data);

FingerState extract_finger_state(const HandLandmarkResult &data);

bool can_classify_hand(const HandLandmarkResult &data);

ClassifierResult classify_hand_gesture(const HandLandmarkResult &data);



}