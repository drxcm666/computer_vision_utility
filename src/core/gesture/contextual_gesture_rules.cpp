#include "cvtool/core/gesture/contextual_gesture_rules.hpp"
#include "cvtool/core/gesture/gesture_rules.hpp"

namespace cvtool::core::gesture
{

static bool is_monkey_gesture(
    const FaceLandmarkResult &face_data, 
    const HandLandmarkResult &hand_data)
{
    cv::Point2f indexTip = hand_data.points[IndexTip];
    cv::Point2f indexMcp = hand_data.points[IndexMcp];
    cv::Point2f mouthCenter = face_data.mouth_center;

    float finger_dist = distance(indexTip, indexMcp);
    float palm_scale = compute_palm_scale(hand_data);
    float mouth_dist = distance(indexTip, mouthCenter);

    bool is_index_bent{(finger_dist / palm_scale) < 1.2};
    bool is_near_mouth{(mouth_dist / face_data.bbox.width) < 0.2};

    if (is_index_bent == true && is_near_mouth == true)
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