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
        return cv::dnn::blobFromImage(
            frame,
            1.0 / 255.0,
            cv::Size(input_width_, input_height_),
            cv::Scalar(),
            true,
            false,
            CV_32F);
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

        if (out_tensor.size() < 3)
            return result;

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

        if (hand_score[0] <= min_hand_score_)
            return result;

        result.confidence = hand_score[0];

        if (lefthand_righthand[0] <= 0.5f)
            result.hand = Handedness::Left;
        else
            result.hand = Handedness::Right;

        float max_abs_xy{0.0f};
        for (int i = 0; i < 21; i++)
        {
            max_abs_xy = std::max(max_abs_xy, std::abs(xyz[i * 3]));
            max_abs_xy = std::max(max_abs_xy, std::abs(xyz[i * 3 + 1]));
        }

        const bool normalized_xy = max_abs_xy <= 1.5f;

        const float target_w = roi.empty() ? static_cast<float>(frame.cols) : static_cast<float>(roi.width);
        const float target_h = roi.empty() ? static_cast<float>(frame.rows) : static_cast<float>(roi.height);
        const float offset_x = roi.empty() ? 0.0f : static_cast<float>(roi.x);
        const float offset_y = roi.empty() ? 0.0f : static_cast<float>(roi.y);

        for (int i = 0; i < 21; i++)
        {
            const float raw_x = xyz[i * 3];
            const float raw_y = xyz[i * 3 + 1];

            float x = 0.0f;
            float y = 0.0f;
            if (normalized_xy)
            {
                x = raw_x * target_w + offset_x;
                y = raw_y * target_h + offset_y;
            }
            else
            {
                x = (raw_x / static_cast<float>(input_width_)) * target_w + offset_x;
                y = (raw_y / static_cast<float>(input_height_)) * target_h + offset_y;
            }

            x = std::clamp(x, 0.0f, static_cast<float>(frame.cols - 1));
            y = std::clamp(y, 0.0f, static_cast<float>(frame.rows - 1));
            result.points[i] = cv::Point2f(x, y);
        }

        result.hand_bbox = cv::boundingRect(result.points);

        const float min_w = target_w * 0.08f;
        const float min_h = target_h * 0.08f;
        const bool too_small_bbox =
            (static_cast<float>(result.hand_bbox.width) < min_w) ||
            (static_cast<float>(result.hand_bbox.height) < min_h);
        if (too_small_bbox && result.confidence < 0.85f)
            return HandLandmarkResult{};

        result.has_hand = true;

        return result;
    }

    cvtool::core::ExitCode HandLandmarkDetector::initialize(
        const std::string &model_path, std::string &err)
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

        session_option_.SetIntraOpNumThreads(2);
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
        HandLandmarkResult result{};

        if (initialized_ == false)
            return result;

        const auto run_inference = [&](const cv::Rect &used_roi) -> HandLandmarkResult
        {
            HandLandmarkResult local_result{};

            cv::Mat local_frame = used_roi.empty() ? frame : frame(used_roi);
            cv::Mat blob = preprocess_image(local_frame);
            Ort::Value input_tensor = create_input_tensor(blob);

            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_.data(),
                &input_tensor, 1,
                output_names_.data(), 3);

            local_result = decode_output(output_tensors, frame, used_roi);
            return local_result;
        };

        try
        {
            if (roi.empty())
            {
                HandLandmarkResult full_frame_result = run_inference(cv::Rect());
                if (full_frame_result.has_hand)
                    tight_miss_streak_ = 0;
                else
                    tight_miss_streak_++;

                return full_frame_result;
            }

            const cv::Rect frame_rect(0, 0, frame.cols, frame.rows);

            const cv::Rect tight_roi = roi & frame_rect;
            if (tight_roi.empty())
                return result;

            const int pad_x = std::max(8, tight_roi.width / 6);
            const int pad_y = std::max(8, tight_roi.height / 6);
            const cv::Rect expanded_raw(
                tight_roi.x - pad_x,
                tight_roi.y - pad_y,
                tight_roi.width + (2 * pad_x),
                tight_roi.height + (2 * pad_y));
            const cv::Rect expanded_roi = expanded_raw & frame_rect;

            HandLandmarkResult tight_result = run_inference(tight_roi);

            if (tight_result.has_hand)
            {
                tight_miss_streak_ = 0;
                return tight_result;
            }

            tight_miss_streak_++;

            constexpr int expanded_after_misses = 2;
            constexpr int expanded_retry_period = 2;
            const bool should_try_expanded =
                tight_miss_streak_ >= expanded_after_misses &&
                (tight_miss_streak_ % expanded_retry_period) == 0;

            if (should_try_expanded && !expanded_roi.empty() && expanded_roi != tight_roi)
            {
                HandLandmarkResult expanded_result = run_inference(expanded_roi);
                if (expanded_result.has_hand)
                {
                    tight_miss_streak_ = 0;
                    return expanded_result;
                }
            }

            return tight_result;
        }
        catch (const Ort::Exception &e)
        {
            last_error_ = e.what();
            return result;
        }
        catch (const std::exception &e)
        {
            last_error_ = e.what();
            return result;
        }
    }

}