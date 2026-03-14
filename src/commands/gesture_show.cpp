#include "cvtool/commands/gesture_show.hpp"
#include "cvtool/core/validate.hpp"
#include "cvtool/core/gesture/gesture_bank.hpp"
#include "cvtool/core/gesture/display_utils.hpp"
#include "cvtool/core/gesture/gesture_domain.hpp"
#include "cvtool/core/gesture/hand_landmark_detector.hpp"
#include "cvtool/core/gesture/gesture_rules.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <vector>
#include <cmath>

static void render_debug_overlay(
    const cvtool::cmd::GestureShowOptions &opt,
    int w, int h, cv::Mat &frame, cv::Rect sr, cvtool::core::gesture::GestureID id,
    float confidence, std::string fingers_str)
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
                fmt::format("gesture: {}", cvtool::core::gesture::to_debug_label(id)),
                cv::Point(x, y + line_height * 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("confidence : {:.2f}", confidence),
                cv::Point(x, y + line_height * 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);

    cv::putText(frame,
                fmt::format("fingers: {}", fingers_str),
                cv::Point(x, y + line_height * 6),
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
    std::string winname = "MyVideo";
    std::string gesture_winname = "Gesture";
    cvtool::core::ExitCode exit_code = cvtool::core::ExitCode::Ok;

    bool window_initialized{false};
    bool roi_warned{false};

    auto current_gesture{cvtool::core::gesture::GestureID::None};

    cvtool::core::gesture::HandLandmarkDetector detector;
    auto det_code = detector.initialize(opt.model_path, err);
    if (det_code != cvtool::core::ExitCode::Ok)
    {
        fmt::println(stderr, "{}", err);
        return det_code;
    }
    std::vector<std::pair<int, int>> connections{
        // Thumb
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
        // Index finger
        {5, 6}, {6, 7}, {7, 8},
        // Middle finger
        {9, 10}, {10, 11}, {11, 12},
        // Ring finger
        {13, 14}, {14, 15}, {15, 16},
        // Pinky finger
        {17, 18}, {18, 19}, {19, 20},
        // Palm (connect the bases of the fingers)
        {0, 5}, {5, 9}, {9, 13}, {13, 17}, {0, 17}};

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
            const int gesture_w = 256;
            const int gesture_h = 200;

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

        cv::Mat display_frame = frame.clone();
        cv::Rect safe_roi{0, 0, 0, 0};
        if (roi_enable)
        {
            cv::Rect scene_rect{0, 0, display_frame.cols, display_frame.rows};
            safe_roi = scene_rect & roi;
            if (!safe_roi.empty())
            {
                roi_warned = false;

                cv::rectangle(display_frame, safe_roi, cv::Scalar(0, 255, 0), 2);
            }
            else if (!roi_warned)
            {
                fmt::println(stderr,
                             "warning: ROI [x={}, y={}, w={}, h={}] is completely outside the camera frame! Ignored.",
                             roi.x, roi.y, roi.width, roi.height);

                roi_warned = true;
            }
        }

        cvtool::core::gesture::HandLandmarkResult result;
        if (opt.roi.empty() || !safe_roi.empty())
        {
            result = detector.detect(frame, safe_roi);
        }  

        std::string debug_fingers_str = "None";
        if (result.has_hand)
        {
            current_gesture = cvtool::core::gesture::classify_hand_gesture(result);
            draw_hand_landmarks(display_frame, result, connections);
            
            auto state = cvtool::core::gesture::extract_finger_state(result);
            
            debug_fingers_str = fmt::format("T={} I={} M={} R={} P={}",
                state.thumb_extended ? 1 : 0, 
                state.index_extended ? 1 : 0, 
                state.middle_extended ? 1 : 0, 
                state.ring_extended ? 1 : 0, 
                state.pinky_extended ? 1 : 0);
        }
        else
        {
            current_gesture = cvtool::core::gesture::GestureID::None;
        }

        const cv::Mat &raw_img = cvtool::core::gesture::get_gesture_image(bank, current_gesture);
        const int gesture_w = 256;
        const int gesture_h = 200;
        cv::Mat display_image = cvtool::core::gesture::letterbox(raw_img, gesture_w, gesture_h);

        if (opt.show_debug)
        {
            render_debug_overlay(
                opt, display_frame.cols, display_frame.rows,
                display_frame, safe_roi,
                current_gesture,
                result.confidence,
                debug_fingers_str);
        }

        cv::imshow(winname, display_frame);
        cv::imshow(gesture_winname, display_image);

        int key{cv::waitKey(30)};
        if (key == 27 || key == 'q' || key == 'Q')
        {
            fmt::println("Stop key pressed by user");
            break;
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
