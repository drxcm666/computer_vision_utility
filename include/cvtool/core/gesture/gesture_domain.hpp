#pragma once 

#include <string_view>

namespace cvtool::core::gesture
{


enum class GestureID
{
    None,
    OpenPalm,
    Fist,
    Peace,
    ThumbsUp,
    Monkey,
    Unknown
};

std::string_view to_asset_key(GestureID id);

std::string_view to_debug_label(GestureID id);



}