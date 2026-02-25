#include "cvtool/core/exit_codes.hpp"
#include "cvtool/commands/info.hpp"
#include "cvtool/commands/gray.hpp"
#include "cvtool/commands/blur.hpp"
#include "cvtool/commands/edges.hpp"
#include "cvtool/commands/video_edges.hpp"
#include "cvtool/commands/contours.hpp"
#include "cvtool/commands/match.hpp"

#include <CLI/CLI.hpp>

#include <filesystem>

namespace Validators
{
    auto odd_or_zero = CLI::Validator(
        [](const std::string &s) -> std::string
        {
            int v{};
            std::size_t pos{};
            try
            {
                v = std::stoi(s, &pos);
            }
            catch (...)
            {
                return "must be an integer";
            }

            if (pos != s.size())
                return "must be an integer";

            // allow 0 (no-op) or odd >= 3
            if (v == 0)
                return {};
            if (v >= 3 && (v % 2 == 1))
                return {};
            return "must be 0 or an odd integer >= 3";
        },
        "ODD_OR_ZERO");

    auto odd_ge_3 = CLI::Validator(
        [](const std::string &s) -> std::string
        {
            int v{};
            std::size_t pos{};
            try
            {
                v = std::stoi(s, &pos);
            }
            catch (...)
            {
                return "must be an integer";
            }

            if (pos != s.size())
                return "must be an integer";

            if (v >= 3 && (v % 2 == 1))
                return {};
            return "must be an odd integer >= 3";
        },
        "ODD_GE_3");

    auto out_path_exist = CLI::Validator(
        [](const std::string &s) -> std::string
        {
            std::filesystem::path p{s};
            if (p.empty())
                return "output path is empty";
            auto dir = p.parent_path();
            if (!dir.empty() && !std::filesystem::exists(dir))
                return "output directory does not exist";
            return {};
        },
        "OUT_DIR_EXISTS");
}

int main(int argc, char **argv)
{
    CLI::App app{"cvtool - console CV utility"};
    app.require_subcommand(1, 1);

    auto *info = app.add_subcommand("info", "Print media metadata");
    auto *gray = app.add_subcommand("gray", "Convert image to grayscale");
    auto *blur = app.add_subcommand("blur", "Blurs the image");
    auto *edges = app.add_subcommand("edges", "Detect edges in image (Canny)");
    auto *video_edges = app.add_subcommand(
        "video-edges", "Detect edges in video frames");
    auto *contours = app.add_subcommand("contours", "Threshold + contour detection + bboxes");
    auto *match = app.add_subcommand("match", "Temple matching (find pattern)");

    cvtool::cmd::InfoOptions inop;
    info->add_option("--in", inop.in_path, "Input file path")->required()->check(CLI::ExistingFile);

    cvtool::cmd::GrayOptions grop;
    gray->add_option("--in", grop.in_path, "Input file path")->required()->check(CLI::ExistingFile);
    gray->add_option("--out", grop.out_path, "Output file path")->required()->check(Validators::out_path_exist);

    cvtool::cmd::BlurOptions blop;
    blur->add_option("--in", blop.in_path, "Input file path")->required()->check(CLI::ExistingFile);
    blur->add_option("--out", blop.out_path, "Output file path")->required()->check(Validators::out_path_exist);
    blur->add_option("--blur-k", blop.blur_k, "Blur coefficient (0 or odd >= 3)")->required()->check(Validators::odd_or_zero);

    cvtool::cmd::EdgesOptions edop;
    edges->add_option("--in", edop.in_path, "Input file path")->required()->check(CLI::ExistingFile);
    edges->add_option("--out", edop.out_path, "Output file path")->required()->check(Validators::out_path_exist);
    edges->add_option("--blur-k", edop.blur_k, "Blur coefficient (0 or odd >= 3)")->required()->check(Validators::odd_or_zero);
    edges->add_option("--low", edop.threshold_low, "Canny lower threshold (0-255)")->required()->check(CLI::Range(0, 255));
    edges->add_option("--high", edop.threshold_high, "Canny upper threshold (0-255)")->required()->check(CLI::Range(0, 255));

    cvtool::cmd::VideoEdgesOptions vept{};
    video_edges->add_option("--in", vept.in_path, "Input file path")->required()->check(CLI::ExistingFile);
    video_edges->add_option("--out", vept.out_path, "Output file path")->required()->check(Validators::out_path_exist);
    video_edges->add_option("--blur-k", vept.blur_k, "Blur coefficient (0 or odd >= 3)")->required()->check(Validators::odd_or_zero);
    video_edges->add_option("--low", vept.low, "Canny lower threshold (0-255)")->required()->check(CLI::Range(0, 255));
    video_edges->add_option("--high", vept.high, "Canny upper threshold (0-255)")->required()->check(CLI::Range(0, 255));
    video_edges->add_option("--every", vept.every, "Process every N-th frame (default: 1)")->check(CLI::Range(1, 1000000));
    video_edges->add_option("--max-frames", vept.max_frames, "Max frames to process (0=all)")->check(CLI::Range(0, 1000000000));
    video_edges->add_option("--codec", vept.codec, "Output codec: auto, mp4v, mjpg, xvid")->check(CLI::IsMember({"auto", "mp4v", "mjpg", "xvid"}));

    cvtool::cmd::ContoursOptions copt{};
    contours->add_option("--in", copt.in_path, "Input image path")->required()->check(CLI::ExistingFile);
    contours->add_option("--out", copt.out_path, "Output image path")->required()->check(Validators::out_path_exist);
    contours->add_option("--thresh", copt.thresh, "otsu|adaptive|manual")->required()->check(CLI::IsMember({"otsu", "adaptive", "manual"}));
    contours->add_option("--blur-k", copt.blur_k, "0 or odd >= 3")->required()->check(Validators::odd_or_zero);
    contours->add_option("--min-area", copt.min_area, "Min area (default: 100.0)")->check(CLI::Range(0.0, 1e18));
    contours->add_option("--draw", copt.draw, "bbox|contour|both")->check(CLI::IsMember({"bbox", "contour", "both"}));
    contours->add_flag("--invert", copt.invert, "Invert mask");
    contours->add_option("--block", copt.block, "Adaptive block (odd > 1)")->check(Validators::odd_ge_3);
    contours->add_option("--c", copt.c, "Adaptive C");
    contours->add_option("--t", copt.t, "Manual threshold 0..255")->check(CLI::Range(0, 255));
    contours->add_option("--json-path", copt.json_path, "Optional JSON report path")->check(Validators::out_path_exist);

    cvtool::cmd::MatchOptions mapt{};
    match->add_option("--in", mapt.in_path, "Input image path")->required()->check(CLI::ExistingFile);
    match->add_option("--out", mapt.out_path, "Output image path")->required()->check(Validators::out_path_exist);
    match->add_option("--templ", mapt.templ_path, "Template image path")->required()->check(CLI::ExistingFile);
    match->add_option("--min-score", mapt.min_score, "Minimal confidence [0...1]")->default_val(0.80);
    match->add_option("--method", mapt.method, "Matching method: ccoeff_normed|ccorr_normed|sqdiff_normed")->default_val("ccoeff_normed");
    match->add_option("--max-results", mapt.max_results, "How many matches to draw (>= 1)")->default_val("5");
    match->add_option("--nms", mapt.nms, "NMS IoU threshold in [0..1] (0.30 default)")->default_val(0.30);
    match->add_option("--mode", mapt.mode, "Match mode: gray|color")->default_val("gray");
    match->add_option("--heatmap", mapt.heatmap_path, "Save heatmap image")->check(Validators::out_path_exist);
    match->add_option("--json", mapt.json_path, "Save JSON report")->check(Validators::out_path_exist);
    match->add_option("--roi", mapt.roi, "ROI: x, y, w, h");
    match->add_option("--draw", mapt.draw, "Draw: bbox|bbox+label|bbox+label+score")->default_val("bbox+label+score");
    match->add_option("--thickness", mapt.thickness, "BBox thickness (>=1)")->default_val(2);
    match->add_option("--font-scale", mapt.font_scale, "Label font scale (>0)")->default_val(0.5);


    cvtool::core::ExitCode rc{0};

    info->callback([&]{ rc = run_info(inop); });

    gray->callback([&]{ rc = run_gray(grop); });

    blur->callback([&]{ rc = run_blur(blop); });

    edges->callback([&]{
        if(edop.threshold_low > edop.threshold_high) {
            throw CLI::ValidationError("--low", "must be < --high");
        }
        rc = run_edges(edop); });

    video_edges->callback([&]{
        if(vept.low > vept.high) {
            throw CLI::ValidationError("--low", "must be < --high");
        }
        rc = run_video_edges(vept); });

    contours->callback([&]{ rc = run_contours(copt); });

    match->callback([&]{ rc = run_match(mapt); });

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e)
    {
        return app.exit(e);
    }

    return to_int(rc);
}
