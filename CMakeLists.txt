cmake_minimum_required(VERSION 3.25)
project(cvtool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio highgui dnn)
find_package(fmt CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create executable
add_executable(cvtool
    src/main.cpp
    src/commands/info.cpp
    src/commands/gray.cpp
    src/commands/blur.cpp
    src/commands/edges.cpp
    src/commands/video_edges.cpp
    src/commands/contours.cpp
    src/commands/match.cpp
    src/commands/match_validate.cpp
    src/commands/gesture_show.cpp

    src/core/edges_pipeline.cpp
    src/core/validate.cpp
    src/core/image_io.cpp
    src/core/video_io.cpp
    src/core/threshold.cpp
    src/core/contours_core.cpp
    src/core/template_match.cpp
    src/core/rois_edges.cpp
    src/core/image_convert.cpp

    src/core/match/match_heatmap.cpp
    src/core/match/match_render.cpp
    src/core/match/match_json.cpp
    src/core/match/match_search_ms.cpp
    src/core/match/match_prepare.cpp

    src/core/gesture/gesture_bank.cpp
    src/core/gesture/display_utils.cpp
    src/core/gesture/gesture_domain.cpp
    src/core/gesture/hand_landmark_detector.cpp
    src/core/gesture/gesture_rules.cpp
    src/core/gesture/gesture_stabilizer.cpp
    src/core/gesture/face_landmark_detector.cpp
    src/core/gesture/contextual_gesture_rules.cpp
)

# Link libraries
target_link_libraries(cvtool PRIVATE
    ${OpenCV_LIBS}
    fmt::fmt
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    onnxruntime::onnxruntime
)

# Include directories
target_include_directories(cvtool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

