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
        float min_hand_score_{0.5f}; // hand detection confidence threshold
        int tight_miss_streak_{0};

        int input_width_{224};
        int input_height_{224};
        std::array<int64_t, 4> input_shape_{1, 3, 224, 224};
        std::array<const char*, 1> input_names_{"input"};
        std::array<const char*, 3> output_names_{"xyz_x21", "hand_score", "lefthand_0_or_righthand_1"};

        cv::Mat preprocess_image(const cv::Mat &frame);
        Ort::Value create_input_tensor(const cv::Mat &blob);
        cvtool::core::gesture::HandLandmarkResult decode_output(
            std::vector<Ort::Value> &out_tensor, const cv::Mat &frame, const cv::Rect &roi);

    public:
        cvtool::core::ExitCode initialize(const std::string &model_path, std::string &err);
        cvtool::core::gesture::HandLandmarkResult detect(const cv::Mat &frame, const cv::Rect &roi = cv::Rect());
    };

}