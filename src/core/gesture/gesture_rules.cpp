#include "cvtool/core/gesture/gesture_rules.hpp"

#include <opencv2/opencv.hpp>

#include <cmath>

namespace cvtool::core::gesture
{

constexpr float min_finger_extension_ratio{0.8f};
constexpr float min_thumb_extension_ratio{0.5f};
constexpr float min_thumb_separation_ratio{0.5f};
constexpr float min_palm_scale{5.0f};

float distance(const cv::Point2f &point1, const cv::Point2f &point2)
{
    float dx = point1.x - point2.x, dy = point1.y - point2.y;
    return std::sqrt(dx * dx + dy * dy);
}

float compute_palm_scale(const cvtool::core::gesture::HandLandmarkResult &data)
{
    return distance(data.points[Wrist], data.points[MiddleMcp]);
}

static bool is_non_thumb_extended(
    const cv::Point2f &mcp, const cv::Point2f &pip, 
    const cv::Point2f &dip, const cv::Point2f &tip, float palm_scale)
{
    if ((tip.y < dip.y && dip.y < pip.y && pip.y < mcp.y) &&
        (distance(tip, mcp) / palm_scale > min_finger_extension_ratio))
        return true;

    return false;
}

static bool is_thumb_extended(
    const cv::Point2f &indexMcp, const cv::Point2f &thumbMcp, 
    const cv::Point2f &thumbTip, float palm_scale)
{
    if ((distance(thumbMcp, thumbTip) / palm_scale > min_thumb_extension_ratio) &&
        distance(thumbTip, indexMcp) / palm_scale > min_thumb_separation_ratio)
        return true;

    return false;
}

static bool is_thumb_pointing_up(
    const cv::Point2f &thumbMcp, const cv::Point2f &thumbIP, const cv::Point2f &thumbTip)
{
    if (thumbTip.y < thumbIP.y && thumbIP.y < thumbMcp.y)
        return true;

    return false;
}

FingerState extract_finger_state(const cvtool::core::gesture::HandLandmarkResult &data)
{
    FingerState state;
    float palm_scale = compute_palm_scale(data);

    state.thumb_extended = is_thumb_extended(
        data.points[IndexMcp], data.points[ThumbMcp], 
        data.points[ThumbTip], palm_scale);

    state.index_extended = is_non_thumb_extended(
        data.points[IndexMcp], data.points[IndexPip], 
        data.points[IndexDip], data.points[IndexTip], palm_scale);

    state.middle_extended = is_non_thumb_extended(
        data.points[MiddleMcp], data.points[MiddlePip], 
        data.points[MiddleDip], data.points[MiddleTip], palm_scale);

    state.ring_extended = is_non_thumb_extended(
        data.points[RingMcp], data.points[RingPip], 
        data.points[RingDip], data.points[RingTip], palm_scale);

    state.pinky_extended = is_non_thumb_extended(
        data.points[PinkyMcp], data.points[PinkyPip], 
        data.points[PinkyDip], data.points[PinkyTip], palm_scale);

    return state;
}

bool can_classify_hand(
    const cvtool::core::gesture::HandLandmarkResult &data)
{
    if (data.has_hand == false || data.confidence < 0.7f)
        return false;

    float palm_scale = compute_palm_scale(data);
    if (palm_scale < min_palm_scale)
        return false;

    return true;
}

cvtool::core::gesture::ClassifierResult classify_hand_gesture(
    const cvtool::core::gesture::HandLandmarkResult &data)
{
    FingerState state = extract_finger_state(data);

    if (state.index_extended == true &&
        state.middle_extended == true &&
        state.ring_extended == true &&
        state.pinky_extended == true &&
        state.thumb_extended == true)
    {
        return {GestureID::OpenPalm, state};
    }

    if (state.index_extended == false &&
        state.middle_extended == false &&
        state.ring_extended == false &&
        state.pinky_extended == false &&
        state.thumb_extended == false)
    {
        return {GestureID::Fist, state};
    }

    if (state.index_extended == true &&
        state.middle_extended == true &&
        state.ring_extended == false &&
        state.pinky_extended == false &&
        state.thumb_extended == false)
    {
        return {GestureID::Peace, state};
    }

    if (state.index_extended == false &&
        state.middle_extended == false &&
        state.ring_extended == false &&
        state.pinky_extended == false &&
        state.thumb_extended == true &&
        is_thumb_pointing_up(data.points[ThumbMcp],
                                data.points[ThumbIp],
                                data.points[ThumbTip]))
    {
        return {GestureID::ThumbsUp, state};
    }

    return {GestureID::Unknown, state};
}

}