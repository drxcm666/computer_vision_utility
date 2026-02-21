#pragma once 

namespace cvtool::core
{

enum class ExitCode : int
{
    Ok = 0,

    InputNotFoundOrNoAccess = 1,
    CannotOpenOrReadInput = 2,
    CannotWriteOutput = 3,

    InvalidParamsOrUnsupported = 4,

    CannotOpenOutputVideo = 5
};

constexpr int to_int(ExitCode c) noexcept
{
    return static_cast<int>(c);
}

}
