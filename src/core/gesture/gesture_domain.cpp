#include "cvtool/core/gesture/gesture_domain.hpp"

namespace cvtool::core::gesture
{

std::string_view to_asset_key(GestureID id)
{
    switch (id)
    {
    case GestureID::None:
        return "none";
    case GestureID::Fist:
        return "fist";
    case GestureID::OpenPalm:
        return "open_palm";
    case GestureID::Peace:
        return "peace";
    case GestureID::ThumbsUp:
        return "thumbs_up";
    default:
        return "unknown";
    }
}

std::string_view to_debug_label(GestureID id)
{
    switch (id)
    {
    case GestureID::None:
        return "None";
    case GestureID::Fist:
        return "Fist (Detected)";
    case GestureID::OpenPalm:
        return "Open Palm (Detected)";
    case GestureID::Peace:
        return "Peace (Detected)";
    case GestureID::ThumbsUp:
        return "Thumbs Up (Detected)";
    default:
        return "Unknown Gesture - Fallback";
    }
}

}