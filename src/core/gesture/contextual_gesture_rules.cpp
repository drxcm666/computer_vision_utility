#include "cvtool/core/gesture/contextual_gesture_rules.hpp"
#include "cvtool/core/gesture/gesture_rules.hpp"

#include <algorithm>

namespace cvtool::core::gesture
{

static bool is_monkey_gesture(
    const FaceLandmarkResult &face_data, 
    const HandLandmarkResult &hand_data)
{
    if (face_data.has_face == false || hand_data.has_hand == false)
        return false;

    if (face_data.confidence < 0.6f)
        return false;

    if (face_data.bbox.width <= 0 || face_data.bbox.height <= 0)
        return false;

    cv::Point2f indexTip = hand_data.points[IndexTip];
    cv::Point2f indexMcp = hand_data.points[IndexMcp];
    cv::Point2f mouthCenter = face_data.mouth_center;

    float finger_dist = distance(indexTip, indexMcp);
    float palm_scale = compute_palm_scale(hand_data);
    if (palm_scale <= 1e-3f)
        return false;

    float mouth_dist = distance(indexTip, mouthCenter);
    const float face_scale = static_cast<float>(std::max(face_data.bbox.width, face_data.bbox.height));
    if (face_scale <= 1e-3f)
        return false;

    const float dx = std::abs(indexTip.x - mouthCenter.x);
    const float dy = std::abs(indexTip.y - mouthCenter.y);

    const float index_ratio = finger_dist / palm_scale;
    const FingerState state = extract_finger_state(hand_data);

    const bool folded_others =
        !state.thumb_extended &&
        !state.middle_extended &&
        !state.ring_extended &&
        !state.pinky_extended;

    bool is_index_slightly_bent{
        !state.index_extended &&
        index_ratio >= 0.55f &&
        index_ratio <= 1.35f};

    bool is_near_mouth{(mouth_dist / face_scale) < 0.20f};
    bool is_locally_near_mouth{
        (dx / static_cast<float>(face_data.bbox.width)) < 0.22f &&
        (dy / static_cast<float>(face_data.bbox.height)) < 0.22f};

    if (folded_others && is_index_slightly_bent && is_near_mouth && is_locally_near_mouth)
        return true;
    
    return false;
}

GestureID classify_contextual_gesture(
    const HandLandmarkResult &hand_data,
    const FaceLandmarkResult &face_data
    )
{
    if(face_data.has_face == false || hand_data.has_hand == false)
        return GestureID::Unknown;

    bool monkey_gesture = is_monkey_gesture(face_data, hand_data);

    if (monkey_gesture)
        return GestureID::Monkey;
    else
        return GestureID::Unknown;
}



}