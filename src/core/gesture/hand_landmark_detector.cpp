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
    cv::resize(result, result, cv::Size(input_width_, input_height_));
    result.convertTo(result, CV_32F, 1.0 / 255.0);

    return result;
}

Ort::Value HandLandmarkDetector::create_input_tensor(const cv::Mat &blob)
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(blob.ptr<float>()), blob.total(), input_shape_.data(), input_shape_.size());

    return input_tensor;
}

cvtool::core::gesture::HandLandmarkResult HandLandmarkDetector::decode_output(
    std::vector<Ort::Value> &out_tensor, const cv::Mat &frame, const cv::Rect &roi)
{
    cvtool::core::gesture::HandLandmarkResult result{};

    if (out_tensor.size() < 3) return result;

    auto info_xyz = out_tensor[0].GetTensorTypeAndShapeInfo();
    std::size_t total_size_xyz = info_xyz.GetElementCount();
    auto info_score = out_tensor[1].GetTensorTypeAndShapeInfo();
    std::size_t total_size_score = info_score.GetElementCount();
    auto info_lr = out_tensor[2].GetTensorTypeAndShapeInfo();
    std::size_t total_size_lr = info_lr.GetElementCount();
    if (total_size_xyz != 63 || total_size_score != 1 || total_size_lr != 1)
        return result;

    
    float *xyz = out_tensor[0].GetTensorMutableData<float>();
    float *hand_score = out_tensor[1].GetTensorMutableData<float>();
    float *lefthand_righthand = out_tensor[2].GetTensorMutableData<float>();

    if (hand_score[0] <= min_hand_score_) return result;

    result.confidence = hand_score[0];

    if (lefthand_righthand[0] <= 0.5f)
        result.hand = Handedness::Left;
    else if (lefthand_righthand[0] > 0.5f)
        result.hand = Handedness::Right;
    else
        result.hand = Handedness::None;

    for (int i = 0; i < 21; i++)
    {
        float x = roi.empty() ? (xyz[i * 3] / static_cast<float>(input_width_) * frame.cols) : 
                                ((xyz[i * 3] / static_cast<float>(input_width_)) * roi.width + roi.x);

        float y = roi.empty() ? (xyz[i * 3 + 1] / static_cast<float>(input_height_) * frame.rows) : 
                                ((xyz[i * 3 + 1] / static_cast<float>(input_height_)) * roi.height + roi.y);

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

    try
    {
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_.data(), 
            &input_tensor, 1, 
            output_names_.data(), 3);
        
        return decode_output(output_tensors, frame, roi);
    }
    catch(const Ort::Exception &e)
    {
        last_error_ = e.what();
        return cvtool::core::gesture::HandLandmarkResult{};
    }
    catch(const std::exception &e)
    {
        last_error_ = e.what();
        return cvtool::core::gesture::HandLandmarkResult{};
    }
}

}