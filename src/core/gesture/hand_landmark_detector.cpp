#include "cvtool/core/gesture/hand_landmark_detector.hpp"

#include <opencv2/dnn.hpp>

#include <fmt/format.h>

#include <array>
#include <filesystem>
#include <iostream>

namespace cvtool::core::gesture
{

cv::Mat HandLandmarkDetector::preprocess_image(const cv::Mat &frame)
{
    cv::Mat result;
    cv::cvtColor(frame, result, cv::COLOR_BGR2RGB);
    cv::resize(result, result, cv::Size(224, 224));
    result.convertTo(result, CV_32F, 1.0 / 255.0);

    return result;
}

Ort::Value HandLandmarkDetector::create_input_tensor(const cv::Mat &blob)
{
    static std::array<int64_t, 4> input_shape{1, 3, 224, 224};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(blob.ptr<float>()), blob.total(), input_shape.data(), input_shape.size());

    return input_tensor;
}

cvtool::core::gesture::HandLandmarkResult HandLandmarkDetector::decode_output(
    std::vector<Ort::Value> &out_tensor, const cv::Mat &frame, const cv::Rect &roi)
{
    cvtool::core::gesture::HandLandmarkResult result{};

    float *xyz = out_tensor[0].GetTensorMutableData<float>();
    float *hand_score = out_tensor[1].GetTensorMutableData<float>();
    float *lefthand_righthand = out_tensor[2].GetTensorMutableData<float>();

    if (hand_score[0] <= 0.5f)
        return result;

    result.confidence = hand_score[0];

    if (lefthand_righthand[0] <= 0.5f)
        result.hand = Handedness::Left;
    else if (lefthand_righthand[0] > 0.5f)
        result.hand = Handedness::Right;
    else
        result.hand = Handedness::None;

    for (int i = 0; i < 21; i++)
    {
        float x = roi.empty() ? (xyz[i * 3] / 224.0f * frame.cols) : 
                                ((xyz[i * 3] / 224.0f) * roi.width + roi.x);

        float y = roi.empty() ? (xyz[i * 3 + 1] / 224.0f * frame.rows) : 
                                ((xyz[i * 3 + 1] / 224.0f) * roi.height + roi.y);

        result.points[i] = cv::Point2f(x, y);
    }

    result.hand_bbox = cv::boundingRect(result.points);

    result.has_hand = true;

    return result;
}

cvtool::core::ExitCode HandLandmarkDetector::initialize(const std::string &model_path, std::string &err)
{
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_FATAL, "Default");

    if (model_path.empty())
    {
        err = fmt::format("error: HandLandmarkDetector::initialize: empty model path");
        initialized_ = false;
        return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
    }

    std::filesystem::path model_fs_path(model_path);
    std::error_code ec;
    const bool model_exists = std::filesystem::exists(model_fs_path, ec);
    if (ec)
    {
        err = fmt::format("error: HandLandmarkDetector::initialize: failed to check model path: {} ({})",
            model_path, ec.message());
        initialized_ = false;
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }
    if (!model_exists)
    {
        err = fmt::format("error: HandLandmarkDetector::initialize: model file does not exist: {}", model_path);
        initialized_ = false;
        return cvtool::core::ExitCode::InputNotFoundOrNoAccess;
    }

    session_option_.SetIntraOpNumThreads(4);
    session_option_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_option_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);
    session_option_.DisableProfiling();

    try
    {
        session_ = std::make_unique<Ort::Session>(*env_, model_fs_path.c_str(), session_option_);
        initialized_ = true;
        return cvtool::core::ExitCode::Ok;
    }
    catch (const Ort::Exception &e)
    {
        err = fmt::format("error: HandLandmarkDetector::initialize: ONNX Runtime failed to create session for model: {}\n"
                          "ORT error: {}",
                          model_path,
                          e.what());
        initialized_ = false;
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }
}

cvtool::core::gesture::HandLandmarkResult HandLandmarkDetector::detect(
    const cv::Mat &frame, const cv::Rect &roi)
{
    if (initialized_ == false)
    {
        return cvtool::core::gesture::HandLandmarkResult{};
    }

    cv::Mat new_frame = roi.empty() ? frame : frame(roi);

    cv::Mat processed_frame = preprocess_image(new_frame);
    cv::Mat blob = cv::dnn::blobFromImage(processed_frame);
    Ort::Value input_tensor = create_input_tensor(blob);

    std::array<const char *, 1> input_names = {"input"};
    std::array<const char *, 3> output_names = {"xyz_x21", "hand_score", "lefthand_0_or_righthand_1"};

    try
    {
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 3);
        
        return decode_output(output_tensors, frame, roi);
    }
    catch(const Ort::Exception &e)
    {
        last_error_ = fmt::format("error: ");
    }
}

}