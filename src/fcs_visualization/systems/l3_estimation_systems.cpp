#include "L3_estimation/tracker/vis_helpers.hpp"
#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"

#include "L3_estimation/tracker/util.hpp"
#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"

#include <Eigen/Core>
#include <cmath>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <string>
#include <vector>

namespace fcs::visualization::foxglove::systems {

[[nodiscard]] inline std::string tracker_image_center_text(double value) {
    return std::isfinite(value) ? fmt::format("{:.1f}", value) : "n/a";
}

[[nodiscard]] inline std::string tracker_label(const ::fcs::L3::TrackerOutput& output) {
    return fmt::format(
        "{}/{} · {}", magic_enum::enum_name(output.target_color),
        magic_enum::enum_name(output.target_name), magic_enum::enum_name(output.status));
}

/// @brief Register L3 estimation layer systems (tracking, association)
///
/// This includes:
/// - foxglove_l3_tracker_scene: Publishes tracker scene visualization
/// - foxglove_l3_association_scene: Publishes association scene (predicted vs measured)
void register_l3_estimation_systems(talos::scheduler::Scheduler& app) {

    using namespace L3::vis;

    // =========================================================================
    // TRACKER VISUALIZATION (L3)
    // =========================================================================

    app.add_system<talos::pool_compute>(
        "foxglove_l3_tracker_scene",
        [](talos::spmc<std::vector<::fcs::L3::TrackerOutput>, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;
            auto outputs = tracker_in.read();
            if (!outputs || outputs->empty())
                return;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Visualize all tracking targets
            for (const auto& output : *outputs) {
                if (!output.is_tracking())
                    continue;

                const std::string target_key =
                    fmt::format("{}_{}", output.target_color, output.target_name);
                const auto pos   = get_tracker_position(output);
                const auto vel   = get_tracker_velocity(output);
                const auto v_yaw = get_tracker_v_yaw(output);

                // Main target sphere (status-colored)
                entities.push_back(
                    viz::patterns::tracker_target<fast_tf::odom>(
                        Eigen::Vector3d(pos.x, pos.y, pos.z), static_cast<int>(output.status),
                        output.timestamp_ns, fmt::format("target_{}", target_key)));

                entities.push_back(
                    viz::EntityBuilder::create<fast_tf::odom>(
                        "l3",
                        fmt::format("tracker_info_{}_{}", output.target_color, output.target_name))
                        .timestamp(output.timestamp_ns)
                        .position(pos)
                        .color(tac::Text::PRIMARY)
                        .text_with_offset(
                            tracker_label(output), tac::L3::TARGET_SIZE * 1.4, 0.0,
                            tac::L3::LABEL_OFFSET_Z, tac::Text::SIZE_SMALL)
                        .metadata(
                            "target_name", std::string(magic_enum::enum_name(output.target_name)))
                        .metadata(
                            "target_color", std::string(magic_enum::enum_name(output.target_color)))
                        .metadata("status", std::string(magic_enum::enum_name(output.status)))
                        .metadata(
                            "last_image_center_distance_px",
                            tracker_image_center_text(output.last_image_center_distance_px))
                        .metadata(
                            "last_observation_timestamp_ns",
                            std::to_string(output.last_observation_timestamp_ns))
                        .build());

                // Velocity arrows
                entities.push_back(
                    viz::patterns::velocity_arrows<fast_tf::odom>(
                        Eigen::Vector3d(pos.x, pos.y, pos.z), vel, v_yaw, output.timestamp_ns,
                        fmt::format("velocity_{}", target_key)));

                // Armor cubes
                if (output.is_robot()) {
                    auto builder = viz::EntityBuilder::create<fast_tf::odom>(
                                       "l3", fmt::format(
                                                 "robot_armors_{}_{}", output.target_color,
                                                 output.target_name))
                                       .timestamp(output.timestamp_ns);

                    add_robot_armor_cubes(builder, *output.robot_state(), output.target_name);
                    entities.push_back(std::move(builder).build());
                } else if (output.is_outpost()) {
                    auto builder = viz::EntityBuilder::create<fast_tf::odom>(
                                       "l3", fmt::format("outpost_armors_{}", output.target_color))
                                       .timestamp(output.timestamp_ns);

                    add_outpost_armor_cubes(builder, *output.outpost_state());
                    entities.push_back(std::move(builder).build());
                }
            }

            publish_scene_if_nonempty<TrackSceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // ASSOCIATION VISUALIZATION (L3) - Predicted vs Measured
    // =========================================================================

    app.add_system<talos::pool_compute>(
        "foxglove_l3_association_scene",
        [](talos::spmc<std::vector<::fcs::L3::TrackerOutput>, TrackerOutputChannelTopic> tracker_in,
           talos::spmc<ArmorMeasurementBatch, MeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;

            auto outputs = tracker_in.read();
            if (!outputs || outputs->empty())
                return;

            auto batch = meas_in.read_current();
            if (!batch)
                return;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            // Predicted armors (present) - neon cyan for all tracking targets
            for (const auto& output : *outputs) {
                if (!output.is_tracking())
                    continue;

                const std::string target_key =
                    fmt::format("{}_{}", output.target_color, output.target_name);
                if (output.is_robot()) {
                    const auto& state = *output.robot_state();
                    const auto armors = state.armor_poses();
                    for (int i = 0; i < 4; i++) {
                        entities.push_back(
                            viz::patterns::predicted_armor<fast_tf::odom>(
                                Eigen::Vector3d(armors[i][0], armors[i][1], armors[i][2]), i,
                                output.timestamp_ns, fmt::format("pred_{}_{}", target_key, i)));
                    }
                } else if (output.is_outpost()) {
                    const auto& state           = *output.outpost_state();
                    constexpr double kAngleStep = 2.0 * std::numbers::pi / 3.0;
                    for (int i = 0; i < 3; i++) {
                        const double armor_yaw = state.yaw + i * kAngleStep;
                        const auto& pos        = state.position;
                        constexpr double r     = L3::OutpostTargetState::radius;
                        const Eigen::Vector3d armor_pos{
                            pos.x() + r * std::cos(armor_yaw), pos.y() + r * std::sin(armor_yaw),
                            state.z[i]};

                        entities.push_back(
                            viz::EntityBuilder::create<fast_tf::odom>(
                                "l3", fmt::format("pred_{}_{}", target_key, i))
                                .position(armor_pos)
                                .size(tac::L3::ARMOR_PLATE_SIZE)
                                .color(tac::L3::TRACKING_LOCKED)
                                .sphere()
                                .text_with_offset(
                                    fmt::format("{}:{}", target_key, i), 0, 0,
                                    tac::L3::LABEL_OFFSET_Z)
                                .timestamp(output.timestamp_ns)
                                .build());
                    }
                }
            }

            // Measurements (past) - ghost silver
            for (const auto& m : batch->measurements) {
                const auto t = m.transform.translation();
                entities.push_back(
                    viz::patterns::measurement_sphere<fast_tf::odom>(
                        Eigen::Vector3d(t.x(), t.y(), t.z()), m.confidence, m.timestamp_ns));
            }

            publish_scene_if_nonempty<AssociationSceneMessage>(*server, std::move(entities));
        });

    // =========================================================================
    // EKF HEATMAP VISUALIZATION (L3 Debug)
    // =========================================================================
    // Renders P (covariance) and K (Kalman gain) as false-color heatmap images.
    // Each tracking target gets a panel, composited into one JPEG.

    app.add_system<talos::pool_compute>(
        "foxglove_l3_ekf_heatmap",
        [](talos::spmc<std::vector<::fcs::L3::TrackerOutput>, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;
            auto outputs = tracker_in.read();
            if (!outputs || outputs->empty())
                return;

            // Filter to tracking targets with valid matrices
            std::vector<const ::fcs::L3::TrackerOutput*> active;
            for (const auto& o : *outputs) {
                if (o.is_tracking())
                    active.push_back(&o);
            }
            if (active.empty())
                return;

            // --- Heatmap renderer with axis labels ---
            const auto render_heatmap =
                [](const Eigen::MatrixXd& mat, int cell,
                   const std::vector<std::string_view>& row_labels,
                   const std::vector<std::string_view>& col_labels) -> cv::Mat {
                const int rows = static_cast<int>(mat.rows());
                const int cols = static_cast<int>(mat.cols());
                if (rows == 0 || cols == 0)
                    return {};

                // Normalize to [0, 255]
                double vmin  = mat.minCoeff();
                double vmax  = mat.maxCoeff();
                double range = vmax - vmin;
                if (range < 1e-15)
                    range = 1.0;

                cv::Mat gray(rows, cols, CV_8U);
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < cols; ++c)
                        gray.at<uint8_t>(r, c) = static_cast<uint8_t>(
                            std::clamp(255.0 * (mat(r, c) - vmin) / range, 0.0, 255.0));

                cv::Mat colored;
                cv::applyColorMap(gray, colored, cv::COLORMAP_JET);

                // Scale up to cell size
                cv::Mat big;
                cv::resize(
                    colored, big, cv::Size(cols * cell, rows * cell), 0, 0, cv::INTER_NEAREST);

                // Grid lines
                for (int r = 0; r <= rows; ++r)
                    cv::line(big, {0, r * cell}, {cols * cell, r * cell}, {30, 30, 30}, 1);
                for (int c = 0; c <= cols; ++c)
                    cv::line(big, {c * cell, 0}, {c * cell, rows * cell}, {30, 30, 30}, 1);

                // Cell value text
                constexpr double kValFontScale = 0.38;
                const cv::Scalar kValColor(255, 255, 255);
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        const double v = mat(r, c);
                        char buf[32];
                        // Compact format: use scientific for very small/large, fixed otherwise
                        if (std::abs(v) < 1e-2 || std::abs(v) >= 1e4)
                            std::snprintf(buf, sizeof(buf), "%.1e", v);
                        else
                            std::snprintf(buf, sizeof(buf), "%.4f", v);
                        int baseline = 0;
                        cv::Size ts  = cv::getTextSize(
                            buf, cv::FONT_HERSHEY_SIMPLEX, kValFontScale, 1, &baseline);
                        cv::putText(
                            big, buf,
                            cv::Point(
                                c * cell + cell / 2 - ts.width / 2,
                                r * cell + cell / 2 + ts.height / 2),
                            cv::FONT_HERSHEY_SIMPLEX, kValFontScale, kValColor, 1, cv::LINE_AA);
                    }
                }

                // Margins for labels
                const int kLeftMargin = cell + 8;
                const int kTopMargin  = cell + 10;
                const cv::Scalar kLabelColor(210, 210, 210);
                constexpr double kFontScale = 0.45;

                cv::Mat canvas(
                    kTopMargin + big.rows, kLeftMargin + big.cols, CV_8UC3, cv::Scalar(20, 20, 20));
                big.copyTo(canvas(cv::Rect(kLeftMargin, kTopMargin, big.cols, big.rows)));

                // Row labels (left side, right-aligned)
                for (int r = 0; r < rows && r < static_cast<int>(row_labels.size()); ++r) {
                    const std::string label(row_labels[r]);
                    int baseline = 0;
                    cv::Size ts =
                        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, kFontScale, 1, &baseline);
                    cv::putText(
                        canvas, label,
                        cv::Point(
                            kLeftMargin - ts.width - 4,
                            kTopMargin + r * cell + cell / 2 + ts.height / 2),
                        cv::FONT_HERSHEY_SIMPLEX, kFontScale, kLabelColor, 1, cv::LINE_AA);
                }

                // Column labels (top side, centered under each cell)
                for (int c = 0; c < cols && c < static_cast<int>(col_labels.size()); ++c) {
                    const std::string label(col_labels[c]);
                    int baseline = 0;
                    cv::Size ts =
                        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, kFontScale, 1, &baseline);
                    const int cx = kLeftMargin + c * cell + cell / 2;
                    cv::putText(
                        canvas, label, cv::Point(cx - ts.width / 2, kTopMargin - 8),
                        cv::FONT_HERSHEY_SIMPLEX, kFontScale, kLabelColor, 1, cv::LINE_AA);
                }

                return canvas;
            };

            // --- Compose panels per target (vertically stacked) ---
            constexpr int kCell   = 60;
            constexpr int kMargin = 8;
            const cv::Scalar kBg(20, 20, 20);
            const cv::Scalar kWhite(220, 220, 220);
            constexpr int kTitleH = 22;

            // Per-target row: P, K, Q placed horizontally
            struct TargetRow {
                std::string label;
                struct MatPanel {
                    cv::Mat img;
                    std::string tag;
                };
                std::vector<MatPanel> mats;
            };
            std::vector<TargetRow> target_rows;

            // Pre-build label vectors from magic_enum
            const auto make_robo_labels = [] {
                std::vector<std::string_view> labels;
                labels.reserve(L3::STATE_MAX);
                for (uint8_t i = 0; i < L3::STATE_MAX; ++i)
                    labels.push_back(magic_enum::enum_name(static_cast<L3::RoboState>(i)));
                return labels;
            };
            const auto make_outpost_labels = [] {
                std::vector<std::string_view> labels;
                labels.reserve(L3::O_STATE_MAX);
                for (uint8_t i = 0; i < L3::O_STATE_MAX; ++i)
                    labels.push_back(magic_enum::enum_name(static_cast<L3::OutpostState>(i)));
                return labels;
            };
            const auto make_measure_labels = [] {
                std::vector<std::string_view> labels;
                labels.reserve(L3::MEASURE_MAX);
                for (uint8_t i = 0; i < L3::MEASURE_MAX; ++i)
                    labels.push_back(magic_enum::enum_name(static_cast<L3::Measure>(i)));
                return labels;
            };

            static const auto kRoboLabels    = make_robo_labels();
            static const auto kOutpostLabels = make_outpost_labels();
            static const auto kMeasLabels    = make_measure_labels();

            for (const auto* output : active) {
                const std::string key = fmt::format(
                    "{} {}", magic_enum::enum_name(output->target_color),
                    magic_enum::enum_name(output->target_name));

                TargetRow row;
                row.label = key;

                if (output->is_robot()) {
                    const auto& s  = *output->robot_state();
                    const auto& rl = kRoboLabels;
                    if (s.P.size() > 0)
                        row.mats.push_back({render_heatmap(s.P, kCell, rl, rl), "P"});
                    if (s.K.size() > 0)
                        row.mats.push_back({render_heatmap(s.K, kCell, rl, kMeasLabels), "K"});
                    if (s.Q.size() > 0)
                        row.mats.push_back({render_heatmap(s.Q, kCell, rl, rl), "Q"});
                    if (s.R.size() > 0)
                        row.mats.push_back(
                            {render_heatmap(s.R, kCell, kMeasLabels, kMeasLabels), "R"});
                } else if (output->is_outpost()) {
                    const auto& s  = *output->outpost_state();
                    const auto& rl = kOutpostLabels;
                    if (s.P.size() > 0)
                        row.mats.push_back({render_heatmap(s.P, kCell, rl, rl), "P"});
                    if (s.K.size() > 0)
                        row.mats.push_back({render_heatmap(s.K, kCell, rl, kMeasLabels), "K"});
                    if (s.Q.size() > 0)
                        row.mats.push_back({render_heatmap(s.Q, kCell, rl, rl), "Q"});
                    if (s.R.size() > 0)
                        row.mats.push_back(
                            {render_heatmap(s.R, kCell, kMeasLabels, kMeasLabels), "R"});
                }

                if (!row.mats.empty())
                    target_rows.push_back(std::move(row));
            }

            if (target_rows.empty())
                return;

            // Compute composite size
            // Each target row: [title] then [mat_tag + mat_img] horizontally
            // Target rows stacked vertically
            int max_w   = 0;
            int total_h = kMargin;
            for (const auto& tr : target_rows) {
                int row_w     = kMargin;
                int max_mat_h = 0;
                for (const auto& m : tr.mats) {
                    row_w += m.img.cols + kMargin;
                    max_mat_h = std::max(max_mat_h, m.img.rows);
                }
                max_w = std::max(max_w, row_w);
                total_h += 2 * kTitleH + max_mat_h + kMargin;
            }

            cv::Mat composite(total_h, max_w, CV_8UC3, kBg);

            int y_off = kMargin;
            for (const auto& tr : target_rows) {
                // Target label
                cv::putText(
                    composite, tr.label, cv::Point(kMargin, y_off + kTitleH - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, kWhite, 1, cv::LINE_AA);
                y_off += kTitleH;

                // Matrices side by side
                int x_off = kMargin;
                int mat_h = 0;
                for (const auto& m : tr.mats) {
                    // Small tag above matrix
                    cv::putText(
                        composite, m.tag, cv::Point(x_off + 4, y_off + 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 200, 255), 1, cv::LINE_AA);
                    if (!m.img.empty())
                        m.img.copyTo(
                            composite(cv::Rect(x_off, y_off + kTitleH, m.img.cols, m.img.rows)));
                    x_off += m.img.cols + kMargin;
                    mat_h = std::max(mat_h, m.img.rows);
                }
                y_off += kTitleH + mat_h + kMargin;
            }

            // Encode JPEG
            std::vector<uint8_t> compressed;
            if (!cv::imencode(".png", composite, compressed))
                return;

            EkfHeatmapMessage msg;
            msg.payload.timestamp = timestamp_from_ns((*outputs)[0].timestamp_ns);
            msg.payload.frame_id  = "ekf_heatmap";
            msg.payload.format    = "png";
            msg.payload.data      = std::vector<std::byte>(
                reinterpret_cast<const std::byte*>(compressed.data()),
                reinterpret_cast<const std::byte*>(compressed.data() + compressed.size()));
            (*server)->enqueue_message(std::move(msg));
        });
}

} // namespace fcs::visualization::foxglove::systems
