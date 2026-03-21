#include "cvtool/commands/gesture_show.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/gesture/gesture_bank.hpp"
#include "cvtool/core/gesture/display_utils.hpp"
#include "cvtool/core/gesture/gesture_domain.hpp"
#include "cvtool/core/gesture/hand_landmark_detector.hpp"
#include "cvtool/core/gesture/gesture_rules.hpp"
#include "cvtool/core/gesture/gesture_stabilizer.hpp"
#include "cvtool/core/gesture/face_landmark_detector.hpp"
#include "cvtool/core/gesture/contextual_gesture_rules.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <vector>
#include <cmath>

static void render_debug_overlay(
    const cvtool::cmd::GestureShowOptions &opt,
    int w, int h, cv::Mat &frame,
    cv::Rect sr, cvtool::core::gesture::GestureID raw_id,
    cvtool::core::gesture::StabilizerResult &stab_res,
    float confidence, std::string fingers_str,
    const cvtool::core::gesture::FaceLandmarkResult &face_result)
{
    int x{10};
    int y{30};
    int line_height{15};
    cv::Scalar color = cv::Scalar(255, 50, 50);

    cv::putText(frame,
                fmt::format("width: {}", w),
                cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("height: {}", h),
                cv::Point(x, y + line_height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("mirror: {}", opt.mirror),
                cv::Point(x, y + line_height * 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("roi: {}, sr: [x={}, y={}, w={}, h={}]",
                            !opt.roi.empty(), sr.x, sr.y, sr.width, sr.height),
                cv::Point(x, y + line_height * 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("stable gesture: {}",
                            cvtool::core::gesture::to_debug_label(stab_res.stable_gesture)),
                cv::Point(x, y + line_height * 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("raw gesture : {}",
                            cvtool::core::gesture::to_debug_label(raw_id)),
                cv::Point(x, y + line_height * 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("candidate: {} ({}/{})",
                            cvtool::core::gesture::to_debug_label(stab_res.candidate_gesture),
                            stab_res.candidate_count, opt.stable_frames),
                cv::Point(x, y + line_height * 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("confidence : {:.2f}", confidence),
                cv::Point(x, y + line_height * 7),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("fingers: {}", fingers_str),
                cv::Point(x, y + line_height * 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("face confidence: {}", face_result.confidence),
                cv::Point(x, y + line_height * 9),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);
}

static void draw_hand_landmarks(
    cv::Mat &display_frame,
    const cvtool::core::gesture::HandLandmarkResult &result,
    const std::vector<std::pair<int, int>> &connections)
{
    std::array<cv::Point, 21> pixel_points;

    for (int i = 0; i < 21; i++)
    {
        cv::circle(display_frame, cv::Point(result.points[i].x, result.points[i].y), 5, cv::Scalar(0, 255, 0), -1);

        pixel_points[i] = cv::Point(result.points[i].x, result.points[i].y);
    }

    for (auto &bone : connections)
    {
        int indexA = bone.first;
        int indexB = bone.second;

        cv::Point ptA = pixel_points[indexA];
        cv::Point ptB = pixel_points[indexB];

        cv::line(display_frame, ptA, ptB, cv::Scalar(0, 255, 0), 1);
    }
}

cvtool::core::ExitCode run_gesture_show(const cvtool::cmd::GestureShowOptions &opt)
{
    std::string err;
    std::vector<std::string> warnings;

    cv::VideoCapture cap(opt.cam);
    if (!cap.isOpened())
    {
        fmt::println(stderr, "Cannot open the camera");
        return cvtool::core::ExitCode::CannotOpenOrReadInput;
    }
    cap.set(cv::CAP_PROP_FPS, 60);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    int width, height;
    if (!opt.size_str.empty())
    {
        const auto sr_code = cvtool::core::validate::validate_screen_resolution(opt.size_str, width, height, err);
        if (sr_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return sr_code;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }

    cvtool::core::gesture::GestureImageBank bank;
    auto bank_code = cvtool::core::gesture::load_gesture_image_bank(
        opt.map_path, bank, warnings, err);
    if (bank_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return bank_code;
    }
    else
        for (auto &i : warnings)
            fmt::println(stderr, "{}", i);

    cv::Rect roi;
    bool roi_enable{false};
    if (!opt.roi.empty())
    {
        const auto roi_code = cvtool::core::validate::validate_roi(opt.roi, roi, err);
        if (roi_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return roi_code;
        }
        roi_enable = true;
    }

    cv::Mat frame;
    int read_fail_streak{0};
    std::string winname{"MyVideo"};
    std::string gesture_winname{"Gesture"};
    cvtool::core::ExitCode exit_code = cvtool::core::ExitCode::Ok;

    bool window_initialized{false};
    bool roi_warned{false};

    constexpr int gesture_w = 256;
    constexpr int gesture_h = 200;

    int frame_index{0};
    const int hand_infer_interval{2};
    const int face_infer_interval{4};
    cvtool::core::gesture::HandLandmarkResult cached_hand_result{};
    cvtool::core::gesture::FaceLandmarkResult cached_face_result{};
    cvtool::core::gesture::ClassifierResult cached_raw_gesture{cvtool::core::gesture::GestureID::None};

    cvtool::core::gesture::StabilizerResult cached_stab_res{};
    std::string cached_debug_fingers_str{"None"};

    cv::Mat cached_display_image; 
    cvtool::core::gesture::GestureID cached_display_image_gesture{
        cvtool::core::gesture::GestureID::Unknown};

    auto display_gesture{cvtool::core::gesture::GestureID::None};

    cvtool::core::gesture::HandLandmarkDetector hand_detector;
    auto det_code = hand_detector.initialize(opt.hand_model_path, err);
    if (det_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return det_code;
    }
    std::vector<std::pair<int, int>> connections{
        // Thumb
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        // Index finger
        {5, 6},
        {6, 7},
        {7, 8},
        // Middle finger
        {9, 10},
        {10, 11},
        {11, 12},
        // Ring finger
        {13, 14},
        {14, 15},
        {15, 16},
        // Pinky finger
        {17, 18},
        {18, 19},
        {19, 20},
        // Palm (connect the bases of the fingers)
        {0, 5},
        {5, 9},
        {9, 13},
        {13, 17},
        {0, 17}};

    cvtool::core::gesture::GestureStabilizer stabilizer{opt.stable_frames, opt.cooldown_ms};

    cvtool::core::gesture::FaceLandmarkDetector face_detector;
    if (!opt.face_model_path.empty())
    {
        auto face_code = face_detector.initialize(opt.face_model_path, err);
        if (face_code != cvtool::core::ExitCode::Ok)
        {
            fmt::println(stderr, "{}", err);
            return face_code;
        }
    }

    while (true)
    {
        bool bSuccess = cap.read(frame);
        if (!bSuccess)
        {
            read_fail_streak++;
            fmt::println(stderr, "warning: failed to read frame");
            if (read_fail_streak >= 15)
            {
                fmt::println(stderr, "error: camera died");
                exit_code = cvtool::core::ExitCode::CannotOpenOrReadInput;
                break;
            }
            int fail_key = cv::waitKey(100);
            if (fail_key == 27 || fail_key == 'q' || fail_key == 'Q')
            {
                fmt::println("User aborted during camera failure");
                break;
            }
            if (window_initialized)
            {
                if (cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE) < 1.0 ||
                    cv::getWindowProperty(gesture_winname, cv::WND_PROP_VISIBLE) < 1.0)
                {
                    fmt::println("Window closed by user during camera failure");
                    break;
                }
            }
            continue;
        }
        read_fail_streak = 0;

        if (!window_initialized)
        {
            int win_w = frame.cols;
            int win_h = frame.rows;

            cv::namedWindow(winname, cv::WINDOW_NORMAL);
            cv::resizeWindow(winname, win_w, win_h);

            cv::namedWindow(gesture_winname, cv::WINDOW_NORMAL);
            cv::resizeWindow(gesture_winname, gesture_w, gesture_h);

            window_initialized = true;

            if (bank.fallback.empty())
            {
                bank.fallback = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            }
        }

        if (opt.mirror)
            cv::flip(frame, frame, 1);

        cv::Rect safe_roi{0, 0, 0, 0};
        bool has_valid_roi = !roi_enable;
        if (roi_enable)
        {
            cv::Rect scene_rect{0, 0, frame.cols, frame.rows};
            safe_roi = scene_rect & roi;
            if (!safe_roi.empty())
            {
                roi_warned = false;
                has_valid_roi = true;
            }
            else if (!roi_warned)
            {
                fmt::println(stderr,
                             "warning: ROI [x={}, y={}, w={}, h={}] is completely outside the camera frame! Ignored.",
                             roi.x, roi.y, roi.width, roi.height);

                roi_warned = true;
                has_valid_roi = false;
            }
        }

        frame_index++;
        const bool run_hand_now{
            has_valid_roi &&
            (((frame_index % hand_infer_interval) == 1) || !cached_hand_result.has_hand)};

        const bool run_face_now{
            has_valid_roi &&
            (((frame_index % face_infer_interval) == 1) || !cached_face_result.has_face)};

        bool hand_updated_this_frame{false};
        if (has_valid_roi)
        {
            if (run_hand_now)
            {
                cached_hand_result = hand_detector.detect(frame, safe_roi);
                hand_updated_this_frame = true;
            }

            if (opt.enable_contextual_gestures)
            {
                if (run_face_now)
                    cached_face_result = face_detector.detect(frame, safe_roi);
            }
            else
                cached_face_result = {};
        }
        else
        {
            cached_hand_result = {};
            cached_face_result = {};
        }

        if (hand_updated_this_frame)
        {
            if (cvtool::core::gesture::can_classify_hand(cached_hand_result))
            {
                cached_raw_gesture =
                    cvtool::core::gesture::classify_hand_gesture(cached_hand_result);

                if (opt.enable_contextual_gestures)
                {
                    auto ctx_gesture = cvtool::core::gesture::classify_contextual_gesture(
                        cached_hand_result, cached_face_result);

                    if (ctx_gesture != cvtool::core::gesture::GestureID::None &&
                        ctx_gesture != cvtool::core::gesture::GestureID::Unknown &&
                        (cached_raw_gesture.gesture == cvtool::core::gesture::GestureID::Unknown ||
                         cached_raw_gesture.gesture == cvtool::core::gesture::GestureID::Fist))
                    {
                        cached_raw_gesture.gesture = ctx_gesture;
                    }
                }

                cached_stab_res = stabilizer.update(cached_raw_gesture.gesture,
                                                    std::chrono::steady_clock::now());

                cached_debug_fingers_str = fmt::format(
                    "T={} I={} M={} R={} P={}",
                    cached_raw_gesture.state.thumb_extended ? 1 : 0,
                    cached_raw_gesture.state.index_extended ? 1 : 0,
                    cached_raw_gesture.state.middle_extended ? 1 : 0,
                    cached_raw_gesture.state.ring_extended ? 1 : 0,
                    cached_raw_gesture.state.pinky_extended ? 1 : 0);
            }
            else
            {
                cached_raw_gesture = {cvtool::core::gesture::GestureID::None, {}};

                cached_stab_res = stabilizer.update(cvtool::core::gesture::GestureID::None,
                                                    std::chrono::steady_clock::now());

                cached_debug_fingers_str = "None";
            }
        }
        else if (!has_valid_roi)
        {
            cached_raw_gesture = {
                cvtool::core::gesture::GestureID::None, {}};

            cached_stab_res = stabilizer.update(
                cvtool::core::gesture::GestureID::None,
                std::chrono::steady_clock::now());

            cached_debug_fingers_str = "None";
        }

        display_gesture = cached_stab_res.stable_gesture;

        cv::Mat display_frame = frame;

        if (has_valid_roi && roi_enable)
            cv::rectangle(display_frame, safe_roi, cv::Scalar(0, 255, 0), 2);

        if (cvtool::core::gesture::can_classify_hand(cached_hand_result))
            draw_hand_landmarks(display_frame, cached_hand_result, connections);

        if (cached_display_image.empty() ||
            cached_display_image_gesture != display_gesture)
        {
            const cv::Mat &raw_img =
                cvtool::core::gesture::get_gesture_image(bank, display_gesture);

            cached_display_image =
                cvtool::core::gesture::letterbox(raw_img, gesture_w, gesture_h);

            cached_display_image_gesture = display_gesture;
        }

        if (opt.show_debug)
        {
            render_debug_overlay(
                opt, display_frame.cols, display_frame.rows,
                display_frame, safe_roi,
                cached_raw_gesture.gesture,
                cached_stab_res,
                cached_hand_result.confidence,
                cached_debug_fingers_str,
                cached_face_result);

            if (cached_face_result.has_face)
            {
                cv::rectangle(
                    display_frame, cached_face_result.bbox,
                    cv::Scalar(0, 0, 255), 1, 8, 0);
                cv::circle(
                    display_frame, cached_face_result.mouth_center, 10,
                    cv::Scalar(0, 255, 255), 1, 8, 0);
            }
        }

        cv::imshow(winname, display_frame);
        cv::imshow(gesture_winname, cached_display_image);

        int key{cv::waitKey(5)};
        if (key == 27 || key == 'q' || key == 'Q')
        {
            fmt::println("Stop key pressed by user");
            break;
        }
        else if (key == 'r' || key == 'R')
        {
            stabilizer.reset();
        }

        if (cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE) < 1.0 ||
            cv::getWindowProperty(gesture_winname, cv::WND_PROP_VISIBLE) < 1.0)
        {
            fmt::println("Window closed by user (X button)");
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return exit_code;
}
