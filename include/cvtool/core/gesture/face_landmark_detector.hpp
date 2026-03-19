#pragma once 

#include "cvtool/core/exit_codes.hpp"
#include "cvtool/core/gesture/face_landmarks.hpp"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <memory>
#include <array>
#include <vector>

namespace cvtool::core::gesture
{

class FaceLandmarkDetector
{
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions options_;
    std::array<const char*, 1> input_names_{"input.1"};
    std::array<const char *, 9> output_names_{
        "448", "471", "494", "451", "474", "497", "454", "477", "500"};
    bool initialized_{false};
    std::string last_error_;
    float min_face_score_{0.5f};
    int input_width_{640};
    int input_height_{640};
    std::array<int64_t, 4> input_shape_{1, 3, 640, 640};

    cv::Mat preprocess_image(const cv::Mat &frame);
    Ort::Value create_input_tensor(cv::Mat &blob);
    cvtool::core::gesture::FaceLandmarkResult decode_output(
        std::vector<Ort::Value> &out_tensor, const cv::Mat &frame, const cv::Rect &roi
    );


public:
    cvtool::core::ExitCode initialize(const std::string &model_path, std::string &err);

    FaceLandmarkResult detect(const cv::Mat &frame, const cv::Rect &roi = cv::Rect());
};

}