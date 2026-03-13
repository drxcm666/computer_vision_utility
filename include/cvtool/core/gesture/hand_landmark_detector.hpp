#pragma once

#include "cvtool/core/gesture/hand_landmarks.hpp"
#include "cvtool/core/exit_codes.hpp"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <memory>
#include <vector>

namespace cvtool::core::gesture
{

    class HandLandmarkDetector
    {
    private:
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> session_;
        bool initialized_{false};
        Ort::SessionOptions session_option_;
        std::string last_error_{};

        cv::Mat preprocess_image(const cv::Mat &frame);
        Ort::Value create_input_tensor(const cv::Mat &blob);
        cvtool::core::gesture::HandLandmarkResult decode_output(
            std::vector<Ort::Value> &out_tensor, const cv::Mat &frame, const cv::Rect &roi);

    public:
        cvtool::core::ExitCode initialize(const std::string &model_path, std::string &err);
        cvtool::core::gesture::HandLandmarkResult detect(const cv::Mat &frame, const cv::Rect &roi = cv::Rect());
    };

}