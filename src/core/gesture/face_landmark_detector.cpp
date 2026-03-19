#include "cvtool/core/gesture/face_landmark_detector.hpp"

#include <fmt/format.h>

#include <locale>
#include <codecvt>
#include <vector>

namespace cvtool::core::gesture
{

    cv::Mat FaceLandmarkDetector::preprocess_image(const cv::Mat &frame)
    {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(input_width_, input_height_));
        cv::Mat colored;
        cv::cvtColor(resized, colored, cv::COLOR_BGR2RGB);
        cv::Mat result = cv::dnn::blobFromImage(
            colored, 1.0 / 128.0, cv::Size(input_width_, input_height_),
            cv::Scalar(127.5, 127.5, 127.5), false, false);

        return result;
    }

    Ort::Value FaceLandmarkDetector::create_input_tensor(cv::Mat &blob)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        float *data = blob.ptr<float>();
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info, data, blob.total(), input_shape_.data(), input_shape_.size());

        return tensor;
    }

    cvtool::core::ExitCode FaceLandmarkDetector::initialize(
        const std::string &model_path, std::string &err)
    {
        try
        {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_FATAL, "FaceDetectorEnv");
            options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            options_.SetIntraOpNumThreads(2);

            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
            std::wstring w_path = converter.from_bytes(model_path);

            session_ = std::make_unique<Ort::Session>(*env_, w_path.c_str(), options_);

            initialized_ = true;

            return cvtool::core::ExitCode::Ok;
        }
        catch (const Ort::Exception &e)
        {
            err = fmt::format("error: ONNX Runtime issue ({})", e.what());
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
        catch (const std::range_error &e)
        {
            err = fmt::format("error: Invalid UTF-8 in model_path ({})", e.what());
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
        catch (const std::bad_alloc &e)
        {
            err = fmt::format("{}", e.what());
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
        catch (const std::exception &e)
        {
            err = fmt::format("{}", e.what());
            return cvtool::core::ExitCode::InvalidParamsOrUnsupported;
        }
    }

    cvtool::core::gesture::FaceLandmarkResult FaceLandmarkDetector::decode_output(
        std::vector<Ort::Value> &out_tensor, const cv::Mat &frame, const cv::Rect &roi)
    {
        FaceLandmarkResult result;
        const float *score_data = out_tensor[2].GetTensorMutableData<float>();
        auto info = out_tensor[2].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        auto count = info.GetElementCount();

        int best_index{-1};
        float max_score{0.0f};
        for (int i = 0; i < count; i++)
        {
            if (score_data[i] > max_score)
            {
                max_score = score_data[i];
                best_index = i;
            }
        }
        if ((max_score > min_face_score_) && (best_index != -1))
        {
            int cell_index = best_index / 2;
            int row = cell_index / 20;
            int col = cell_index % 20;
            float cc_x = (col * 32.0f) + 16.0f;
            float cc_y = (row * 32.0f) + 16.0f;

            float scale_x = static_cast<float>(frame.cols) / static_cast<float>(input_width_);
            float scale_y = static_cast<float>(frame.rows) / static_cast<float>(input_height_);

            float stride_mult = 32.0f;

            const float *bbox_data = out_tensor[5].GetTensorMutableData<float>();
            float bbox_x_min = cc_x - (bbox_data[best_index * 4 + 0] * stride_mult);
            float bbox_y_min = cc_y - (bbox_data[best_index * 4 + 1] * stride_mult);
            float bbox_x_max = cc_x + (bbox_data[best_index * 4 + 2] * stride_mult);
            float bbox_y_max = cc_y + (bbox_data[best_index * 4 + 3] * stride_mult);

            result.bbox.x = (bbox_x_min * scale_x) + roi.x;
            result.bbox.y = (bbox_y_min * scale_y) + roi.y;
            result.bbox.width = (bbox_x_max - bbox_x_min) * scale_x;
            result.bbox.height = (bbox_y_max - bbox_y_min) * scale_y;

            const float *kps_data = out_tensor[8].GetTensorMutableData<float>();
            float mouth_x_min = cc_x + (kps_data[best_index * 10 + 6] * stride_mult);
            float mouth_y_min = cc_y + (kps_data[best_index * 10 + 7] * stride_mult);
            float mouth_x_max = cc_x + (kps_data[best_index * 10 + 8] * stride_mult);
            float mouth_y_max = cc_y + (kps_data[best_index * 10 + 9] * stride_mult);

            float mouth_cx = (mouth_x_min + mouth_x_max) / 2.0;
            float mouth_cy = (mouth_y_min + mouth_y_max) / 2.0;

            result.mouth_center.x = mouth_cx * scale_x + roi.x;
            result.mouth_center.y = mouth_cy * scale_y + roi.y;


            result.confidence = max_score;
            result.has_face = true;
        }

        return result;
    }

    FaceLandmarkResult FaceLandmarkDetector::detect(
        const cv::Mat &frame, const cv::Rect &roi)
    {
        FaceLandmarkResult result;

        if (!initialized_)
            return result;

        cv::Mat new_frame = roi.empty() ? frame : frame(roi);

        cv::Mat blob = preprocess_image(new_frame);
        Ort::Value input_tensor = create_input_tensor(blob);
        std::vector<Ort::Value> output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor, 1,
            output_names_.data(),
            output_names_.size());

        return decode_output(output_tensors, new_frame, roi);
    }

}