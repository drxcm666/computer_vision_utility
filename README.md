# cvtool

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4-green?logo=opencv&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.19-grey?logo=onnx&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows-0078D4?logo=windows&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.25+-064F8C?logo=cmake&logoColor=white)
![vcpkg](https://img.shields.io/badge/vcpkg-enabled-blueviolet)
![Build System](https://img.shields.io/badge/build-Ninja-orange)

A command-line computer vision utility written in **C++20** with OpenCV, ONNX Runtime, and a modular pipeline architecture.

Supported operations: image/video metadata inspection, grayscale conversion, Gaussian blur, Canny edge detection (images and video), contour detection, multi-scale template matching, and real-time webcam gesture recognition with optional face-context fusion.

---

## Table of Contents

- [Requirements](#requirements)
- [Building](#building)
- [Commands](#commands)
  - [info](#info)
  - [gray](#gray)
  - [blur](#blur)
  - [edges](#edges)
  - [video-edges](#video-edges)
  - [contours](#contours)
  - [match](#match)
  - [gesture-show](#gesture-show)
- [Gesture Map JSON](#gesture-map-json)
- [Exit Codes](#exit-codes)
- [Project Structure](#project-structure)

---

## Screenshots

### Gesture Recognition

| Peace ✌️ | Thumbs Up 👍 |
|:---:|:---:|
| ![Peace gesture](<img width="853" height="510" alt="image" src="https://github.com/user-attachments/assets/3bf85718-3901-455e-9b02-218dc9ffa15d" />
) | ![Thumbs Up gesture](<img width="843" height="507" alt="image" src="https://github.com/user-attachments/assets/5a1b07f2-3d18-418d-bf68-a7f660c3ed4c" />
) |
| Index + middle extended, others folded | Thumb up, all other fingers closed |

> Screenshots taken with `--show-debug` and `--mirror` flags enabled.

### Template Matching — Car Detection in a Parking Lot

![Car detection result](docs/screenshots/match_car_parking.png)

> `cvtool match` found the car on the parking lot image using multi-scale template matching.  
> Green bounding box with confidence score and scale label.

```bash
cvtool match \
  --in  docs/screenshots/parking_scene.jpg \
  --out docs/screenshots/match_car_parking.png \
  --templ docs/screenshots/car_template.jpg \
  --scales 0.6:1.4:0.05 \
  --min-score 0.72 \
  --max-results 1 \
  --draw bbox+label+score \
  --heatmap docs/screenshots/match_car_heatmap.png
```

> **Як додати скріншоти:**
> 1. Створи папку `docs/screenshots/` в корені проекту
> 2. Запусти команди вище зі своїми зображеннями
> 3. Збережи результати за вказаними шляхами — GitHub відрендерить їх автоматично

---

## Requirements

| Tool / Library | Version |
|---|---|
| C++ compiler | GCC 13+ (MinGW64 or UCRT64) with C++20 |
| CMake | 3.25+ |
| Ninja | any recent |
| vcpkg | latest |
| OpenCV 4 | with `dnn` and `ffmpeg` features |
| ONNX Runtime | via vcpkg |
| fmt | via vcpkg |
| CLI11 | via vcpkg |
| nlohmann-json | via vcpkg |

> **Note:** The project includes custom vcpkg overlays for ONNX (`vcpkg-overlays/onnx/`) that apply patches required for correct MinGW builds. These are applied automatically during the vcpkg install step.

---

## Building

### 1. Copy and fill in the CMake preset

```bash
cp "CMakePresets.template - Copy.json" CMakePresets.json
```

Edit `CMakePresets.json` and replace the placeholder paths with your actual paths to `gcc`, `g++`, `ninja`, and your `vcpkg` installation.

### 2. Configure and build

```bash
# MinGW64
cmake --preset debug-ninja-mingw64-vcpkg
cmake --build --preset debug-mingw64

# or UCRT64
cmake --preset debug-ninja-ucrt64-vcpkg
cmake --build --preset debug-ucrt64
```

The binary will be placed in `build-ninja-mingw64/` or `build-ninja-ucrt64/` respectively.

---

## Commands

All commands follow this pattern:

```
cvtool <command> [options]
```

Run `cvtool --help` or `cvtool <command> --help` for full option details.

---

### info

Inspect metadata of an image or video file.

```
cvtool info --in <path>
```

**Options:**

| Option | Description |
|---|---|
| `--in <path>` | Input file (image or video). Required. |

**Example:**

```bash
cvtool info --in photo.jpg
```

**Output (image):**
```
kind: image
path: photo.jpg
size: 1920x1080
channels: 3
depth: 8u (0)
mat_type: CV_8UC3
```

**Output (video):**
```
kind: video
path: clip.mp4
size: 1280x720
fps: 30.00
frames: 900
duration_s: 30.00
```

---

### gray

Convert an image to grayscale. Supports 1-, 3-, and 4-channel 8-bit images.

```
cvtool gray --in <path> --out <path>
```

**Options:**

| Option | Description |
|---|---|
| `--in <path>` | Input image. Required. |
| `--out <path>` | Output image. Required. |

**Example:**

```bash
cvtool gray --in photo.jpg --out gray.jpg
```

---

### blur

Apply Gaussian blur to an image.

```
cvtool blur --in <path> --out <path> --blur-k <k>
```

**Options:**

| Option | Description |
|---|---|
| `--in <path>` | Input image. Required. |
| `--out <path>` | Output image. Required. |
| `--blur-k <k>` | Kernel size. Must be `0` (no blur) or an odd integer `>= 3`. Required. |

**Example:**

```bash
cvtool blur --in photo.jpg --out blurred.jpg --blur-k 7
```

---

### edges

Detect edges in a single image using the Canny algorithm.

```
cvtool edges --in <path> --out <path> --blur-k <k> --low <n> --high <n>
```

**Options:**

| Option | Description |
|---|---|
| `--in <path>` | Input image. Required. |
| `--out <path>` | Output image (grayscale edge map). Required. |
| `--blur-k <k>` | Pre-blur kernel (`0` or odd `>= 3`). Required. |
| `--low <n>` | Canny lower threshold `[0..255]`. Required. |
| `--high <n>` | Canny upper threshold `[0..255]`, must be `> low`. Required. |

**Example:**

```bash
cvtool edges --in photo.jpg --out edges.png --blur-k 5 --low 50 --high 150
```

---

### video-edges

Process a video file frame-by-frame and write an edge-detected output video.

```
cvtool video-edges --in <path> --out <path> --blur-k <k> --low <n> --high <n> [options]
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--in <path>` | — | Input video. Required. |
| `--out <path>` | — | Output video. Required. |
| `--blur-k <k>` | — | Pre-blur kernel (`0` or odd `>= 3`). Required. |
| `--low <n>` | — | Canny lower threshold. Required. |
| `--high <n>` | — | Canny upper threshold. Required. |
| `--every <n>` | `1` | Process every N-th frame. |
| `--max-frames <n>` | `0` | Maximum frames to process (`0` = all). |
| `--codec <name>` | `auto` | Output codec: `auto`, `mp4v`, `mjpg`, `xvid`. |

> With `--codec auto`, the codec is selected by the output file extension: `.mp4` → `mp4v`, `.avi` → `xvid`, otherwise `mjpg`.

**Example:**

```bash
cvtool video-edges --in clip.mp4 --out edges.mp4 --blur-k 5 --low 50 --high 150 --every 2
```

---

### contours

Convert an image to a binary mask and detect contours with bounding boxes. Optionally exports a JSON report.

```
cvtool contours --in <path> --out <path> --thresh <mode> --blur-k <k> [options]
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--in <path>` | — | Input image. Required. |
| `--out <path>` | — | Annotated output image. Required. |
| `--thresh <mode>` | — | Threshold mode: `otsu`, `adaptive`, `manual`. Required. |
| `--blur-k <k>` | — | Pre-blur kernel. Required. |
| `--min-area <n>` | `100.0` | Minimum contour area in pixels to keep. |
| `--draw <mode>` | `bbox` | Annotation mode: `bbox`, `contour`, `both`. |
| `--invert` | `false` | Invert the binary mask. |
| `--block <n>` | `11` | Adaptive threshold block size (odd, `>= 3`). Used with `--thresh adaptive`. |
| `--c <n>` | `2.0` | Adaptive threshold C constant. |
| `--t <n>` | `–1` | Manual threshold value `[0..255]`. Used with `--thresh manual`. |
| `--json-path <path>` | — | Optional path to write a JSON report (up to 200 contours). |

**Example:**

```bash
cvtool contours --in photo.jpg --out annotated.jpg \
  --thresh otsu --blur-k 5 --min-area 200 --draw both --json-path report.json
```

**JSON report structure:**

```json
{
  "command": "contours",
  "input": "photo.jpg",
  "output": "annotated.jpg",
  "threshold": { "mode": "otsu", "blur_k": 5, "invert": false, "params": {} },
  "stats": {
    "contours_total": 42,
    "contours_kept": 17,
    "area_min": 210.5,
    "area_mean": 850.3,
    "area_max": 4200.0
  },
  "items_truncated": false,
  "items": [
    { "id": 0, "area": 4200.0, "bbox": { "x": 10, "y": 20, "w": 80, "h": 60 } }
  ]
}
```

---

### match

Multi-scale template matching with NMS, optional ROI auto-detection, heatmap output, and JSON report.

```
cvtool match --in <path> --out <path> --templ <path> [options]
```

**Core options:**

| Option | Default | Description |
|---|---|---|
| `--in <path>` | — | Scene image. Required. |
| `--out <path>` | — | Annotated output image. Required. |
| `--templ <path>` | — | Template image. Required. |
| `--method <name>` | `ccoeff_normed` | Matching method: `ccoeff_normed`, `ccorr_normed`, `sqdiff_normed`. |
| `--mode <name>` | `gray` | Color mode for matching: `gray`, `color`. |
| `--min-score <f>` | `0.80` | Minimum confidence threshold `[0..1]`. |
| `--max-results <n>` | `5` | Maximum number of results to keep after NMS. |
| `--nms <f>` | `0.30` | NMS IoU threshold `[0..1]`. |
| `--scales <min:max:step>` | `1.0:1.0:0.05` | Multi-scale range (e.g. `0.5:1.5:0.1`). |
| `--per-scale-top <n>` | auto | Candidates collected per scale before NMS. |
| `--max-scales <n>` | `0` | Hard limit on the number of scales (`0` = unlimited). |

**ROI options:**

| Option | Default | Description |
|---|---|---|
| `--roi <x,y,w,h>` | — | Manual region of interest. |
| `--roi-auto <mode>` | `none` | Auto ROI detection: `none`, `edges`. |
| `--roi-max <n>` | `8` | Max number of auto-detected ROIs. |
| `--roi-min-area <f>` | `0.01` | Min ROI area as fraction of scene area. |
| `--roi-pad <n>` | `10` | Padding around detected ROI (pixels). |
| `--roi-merge-iou <f>` | `0.20` | Merge ROIs with IoU above this threshold. |
| `--roi-fallback` | `true` | Fall back to full scene if no ROI is found. |
| `--roi-edges-low <n>` | `60` | Canny low threshold for edge-based ROI. |
| `--roi-edges-high <n>` | `140` | Canny high threshold for edge-based ROI. |
| `--roi-edges-blur-k <k>` | `5` | Pre-blur for edge-based ROI. |
| `--draw-roi` | `false` | Draw detected ROI rectangles on output. |

**Output options:**

| Option | Default | Description |
|---|---|---|
| `--draw <mode>` | `bbox+label+score` | Draw mode: `bbox`, `bbox+label`, `bbox+label+score`. |
| `--thickness <n>` | `2` | Bounding box line thickness. |
| `--font-scale <f>` | `0.5` | Label font scale. |
| `--heatmap <path>` | — | Write a colorized heatmap of the match response. |
| `--json <path>` | — | Write a JSON report. |

**Example:**

```bash
cvtool match --in scene.jpg --out result.jpg --templ logo.png \
  --scales 0.7:1.3:0.05 --min-score 0.75 --max-results 3 \
  --roi-auto edges --heatmap heat.png --json matches.json
```

---

### gesture-show

Real-time webcam gesture recognition. Displays detected gestures in a separate window. Optionally enables face-context-aware gestures (e.g. `Monkey` — finger near mouth).

```
cvtool gesture-show --map <path> --model <path> [options]
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--map <path>` | — | Path to the gesture image map JSON. Required. |
| `--model <path>` | — | Path to the ONNX hand landmark model. Required. |
| `--cam <n>` | `0` | Camera device index. |
| `--size <WxH>` | — | Camera capture resolution (e.g. `1280x720`). |
| `--mirror` | `false` | Mirror the camera image horizontally. |
| `--roi <x,y,w,h>` | — | Region of interest within the frame. |
| `--show-debug` | `false` | Show debug overlay (dimensions, gesture state, finger flags, confidence, face data). |
| `--stable-frames <n>` | `5` | Consecutive frames required to confirm a gesture change. |
| `--cooldown-ms <n>` | `300` | Minimum milliseconds between gesture state changes. |
| `--face-model <path>` | — | Path to ONNX face landmark model. Enables face data for contextual gestures. |
| `--face-conf <f>` | `0.5` | Minimum confidence threshold for face detection. |
| `--contextual-gestures` | `false` | Enable contextual gesture classifier (requires `--face-model`). |

**Keyboard controls (in the video window):**

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `R` | Reset gesture stabilizer |
| Close window | Quit |

**Recognized gestures:**

| Gesture | Asset key | Description |
|---|---|---|
| `None` | `none` | No hand detected |
| `OpenPalm` | `open_palm` | All five fingers extended |
| `Fist` | `fist` | All fingers closed |
| `Peace` | `peace` | Index and middle extended, others closed |
| `ThumbsUp` | `thumbs_up` | Thumb extended and pointing up, others closed |
| `Monkey` | `monkey` | Index bent near mouth (contextual, requires face model) |
| `Unknown` | `unknown` | Hand detected but gesture not recognized |

**Example:**

```bash
cvtool gesture-show \
  --map gestures/map.json \
  --model models/hand_landmark.onnx \
  --face-model models/face_detector.onnx \
  --contextual-gestures \
  --mirror --show-debug --stable-frames 6 --cooldown-ms 400
```

---

## Gesture Map JSON

The `--map` file maps gesture asset keys to image paths (absolute or relative to the JSON file).

```json
{
  "none":       "images/none.png",
  "open_palm":  "images/open_palm.png",
  "fist":       "images/fist.png",
  "peace":      "images/peace.png",
  "thumbs_up":  "images/thumbs_up.png",
  "monkey":     "images/monkey.png",
  "unknown":    "images/unknown.png"
}
```

- The `none` entry also serves as the **fallback image** when no gesture image is found for a given state.
- Unknown keys are ignored with a warning.
- Missing image files produce a warning but do not abort startup.

---

## Exit Codes

| Code | Constant | Meaning |
|---|---|---|
| `0` | `Ok` | Success |
| `1` | `InputNotFoundOrNoAccess` | Input file not found or inaccessible |
| `2` | `CannotOpenOrReadInput` | File exists but cannot be opened or decoded |
| `3` | `CannotWriteOutput` | Cannot write to the output path |
| `4` | `InvalidParamsOrUnsupported` | Invalid parameters or unsupported operation |
| `5` | `CannotOpenOutputVideo` | Cannot open or create output video writer |

---

## Project Structure

```
cvtool/
├── include/
│   └── cvtool/
│       ├── commands/          # Option structs and command entry-point declarations
│       │   ├── blur.hpp
│       │   ├── contours.hpp
│       │   ├── edges.hpp
│       │   ├── gesture_show.hpp
│       │   ├── gray.hpp
│       │   ├── info.hpp
│       │   ├── match.hpp
│       │   ├── match_validate.hpp
│       │   └── video_edges.hpp
│       └── core/
│           ├── exit_codes.hpp
│           ├── validate.hpp
│           ├── image_io.hpp
│           ├── image_convert.hpp
│           ├── video_io.hpp
│           ├── edges_pipeline.hpp
│           ├── threshold.hpp
│           ├── contours_core.hpp
│           ├── template_match.hpp
│           ├── match_types.hpp
│           ├── rois_edges.hpp
│           ├── match/
│           │   ├── match_heatmap.hpp
│           │   ├── match_json.hpp
│           │   ├── match_prepare.hpp
│           │   ├── match_render.hpp
│           │   └── match_search_ms.hpp
│           └── gesture/
│               ├── gesture_domain.hpp       # GestureID enum, to_asset_key, to_debug_label
│               ├── hand_landmarks.hpp       # HandLandmarkResult struct
│               ├── face_landmarks.hpp       # FaceLandmarkResult struct
│               ├── gesture_rules.hpp        # FingerState, ClassifierResult, classify_hand_gesture
│               ├── contextual_gesture_rules.hpp  # classify_contextual_gesture
│               ├── gesture_stabilizer.hpp   # GestureStabilizer (hysteresis + cooldown)
│               ├── gesture_bank.hpp         # Image bank loader
│               ├── hand_landmark_detector.hpp
│               ├── face_landmark_detector.hpp
│               └── display_utils.hpp        # letterbox()
├── src/
│   ├── main.cpp               # CLI11 wiring for all subcommands
│   ├── commands/              # Command implementations
│   └── core/                  # Pipeline and utility implementations
├── vcpkg-overlays/onnx/       # Custom vcpkg port for ONNX with MinGW patches
├── vcpkg-triplets/            # Custom vcpkg triplets
├── vcpkg.json                 # vcpkg manifest
├── CMakeLists.txt
└── CMakePresets.template.json
```
