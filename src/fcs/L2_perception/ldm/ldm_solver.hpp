#pragma once

#include "L2_perception/ldm/ldm_geometry.hpp"
#include "L2_perception/ldm/types.hpp"
#include "camera_config.hpp"
#include "frame.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace fcs::L2::ldm {

enum class PnPError {
    InvalidDetection,
    NoCandidate,
};

class LdmSolver {
public:
    using OdomCameraTransform = fast_tf::FrameTransform<fast_tf::odom, fast_tf::camera_optical>;

    explicit LdmSolver(
        const CameraConfig& config, const LdmDetectorConfig& detector_config) noexcept
        : config_(detector_config) {
        cv::eigen2cv(config.camera_matrix, camera_matrix_);
        cv::eigen2cv(config.distort_coefficient, dist_coeffs_);
    }

    [[nodiscard]] std::expected<LdmMeasurement, PnPError>
        solve(const LdmDetection& detection, const OdomCameraTransform& T_odom_camera) const {
        if (detection.pairs.empty()) {
            return std::unexpected(PnPError::InvalidDetection);
        }

        auto preliminary_candidates = detection.mesh_candidates;
        if (preliminary_candidates.empty()) {
            preliminary_candidates = build_preliminary_mesh_candidates(detection.pairs, config_);
        }
        if (preliminary_candidates.empty()) {
            return std::unexpected(PnPError::NoCandidate);
        }

        std::vector<LdmMeshCandidate> evaluated_candidates;
        evaluated_candidates.reserve(preliminary_candidates.size() * 16);

        for (const auto& base_candidate : preliminary_candidates) {
            if (base_candidate.pair_indices.empty()) {
                continue;
            }

            if (has_face_assignment(base_candidate)) {
                auto candidate = base_candidate;
                evaluate_candidate(candidate, detection.pairs, T_odom_camera);
                evaluated_candidates.push_back(std::move(candidate));
                continue;
            }

            for (int start_face = 0; start_face < 8; ++start_face) {
                for (const bool reverse : {false, true}) {
                    auto candidate = base_candidate;
                    candidate.octagon_face_indices =
                        make_face_assignment(start_face, candidate.pair_indices.size(), reverse);
                    candidate.reverse_assignment = reverse;
                    evaluate_candidate(candidate, detection.pairs, T_odom_camera);
                    evaluated_candidates.push_back(std::move(candidate));
                }
            }
        }

        if (evaluated_candidates.empty()) {
            return std::unexpected(PnPError::NoCandidate);
        }

        auto combined_candidates =
            build_combined_candidates(evaluated_candidates, detection.pairs, T_odom_camera);
        evaluated_candidates.insert(
            evaluated_candidates.end(), combined_candidates.begin(), combined_candidates.end());

        std::optional<size_t> best_depth_idx;
        for (size_t i = 0; i < evaluated_candidates.size(); ++i) {
            const auto& candidate = evaluated_candidates[i];
            if (candidate.depth_valid
                && (!best_depth_idx.has_value()
                    || candidate.score > evaluated_candidates[*best_depth_idx].score)) {
                best_depth_idx = i;
            }
        }
        if (!best_depth_idx.has_value()) {
            return std::unexpected(PnPError::NoCandidate);
        }

        const size_t selected_idx = *best_depth_idx;
        const auto selected       = evaluated_candidates[selected_idx];

        LdmMeasurement measurement;
        measurement.timestamp_ns        = detection.timestamp_ns;
        measurement.frame_id            = detection.frame_id;
        measurement.color               = detection.color;
        measurement.pair_count_total    = static_cast<int>(detection.pairs.size());
        measurement.selected_pair_count = static_cast<int>(selected.pair_indices.size());
        measurement.center_image_px     = selected.estimated_center_image_px;
        measurement.bearing_cam     = bearing_for_image_point(selected.estimated_center_image_px);
        measurement.mesh_candidates = std::move(evaluated_candidates);
        measurement.selected_candidate_idx = static_cast<int>(selected_idx);

        measurement.transform_cam  = selected.pose.camera;
        measurement.transform_odom = selected.pose.odom;
        measurement.depth_quality  = (selected.pair_indices.size() >= 3)
                                       ? LdmDepthQuality::Stable
                                       : LdmDepthQuality::Constrained;

        measurement.confidence = confidence_for(selected);
        return measurement;
    }

private:
    struct PoseSeed {
        double yaw{0.0};
        Eigen::Vector3d translation{0.0, 0.0, 1.0};
        double normalized_rmse{std::numeric_limits<double>::infinity()};
    };

    struct CandidateCorrespondences {
        std::vector<cv::Point3f> obj_points_cv{};
        std::vector<cv::Point2f> img_points_cv{};
        std::vector<Eigen::Vector3d> obj_points{};
        std::vector<Eigen::Vector2d> normalized_points{};
    };

    struct ConstrainedPoseSolution {
        Eigen::Matrix3d rotation_cam{Eigen::Matrix3d::Identity()};
        Eigen::Vector3d center_cam{0.0, 0.0, 1.0};
        cv::Mat rvec{};
        cv::Mat tvec{};
        float reprojection_rmse_px{std::numeric_limits<float>::infinity()};
        float vertical_axis_term{0.0f};
        float face_visibility_term{0.0f};
        cv::Point2f projected_center_image{};
        std::vector<cv::Point2f> projected_outline_image{};
    };

    struct NormalizedReprojectionError {
        NormalizedReprojectionError(
            const Eigen::Vector3d& model_point, const Eigen::Vector2d& normalized_point)
            : model_x(model_point.x())
            , model_y(model_point.y())
            , model_z(model_point.z())
            , observed_x(normalized_point.x())
            , observed_y(normalized_point.y()) {}

        template <typename T>
        bool operator()(const T* const yaw, const T* const translation, T* residuals) const {
            const T cy = ceres::cos(yaw[0]);
            const T sy = ceres::sin(yaw[0]);

            const T x = cy * T(model_x) + sy * T(model_z) + translation[0];
            const T y = T(model_y) + translation[1];
            const T z = -sy * T(model_x) + cy * T(model_z) + translation[2];

            residuals[0] = x / z - T(observed_x);
            residuals[1] = y / z - T(observed_y);
            return true;
        }

        [[nodiscard]] static ceres::CostFunction*
            Create(const Eigen::Vector3d& model_point, const Eigen::Vector2d& normalized_point) {
            return new ceres::AutoDiffCostFunction<NormalizedReprojectionError, 2, 1, 3>(
                new NormalizedReprojectionError(model_point, normalized_point));
        }

        double model_x{0.0};
        double model_y{0.0};
        double model_z{0.0};
        double observed_x{0.0};
        double observed_y{0.0};
    };

    [[nodiscard]] std::vector<int>
        make_face_assignment(int start_face, size_t pair_count, bool reverse) const {
        std::vector<int> face_indices;
        face_indices.reserve(pair_count);
        for (size_t i = 0; i < pair_count; ++i) {
            int face = 0;
            if (reverse) {
                face = (start_face - static_cast<int>(i)) % 8;
            } else {
                face = (start_face + static_cast<int>(i)) % 8;
            }
            if (face < 0) {
                face += 8;
            }
            face_indices.push_back(face);
        }
        return face_indices;
    }

    [[nodiscard]] bool has_face_assignment(const LdmMeshCandidate& candidate) const noexcept {
        return !candidate.pair_indices.empty()
            && candidate.pair_indices.size() == candidate.octagon_face_indices.size();
    }

    [[nodiscard]] bool face_sequence_is_consistent(
        const std::vector<int>& face_indices, bool reverse) const noexcept {
        if (face_indices.size() <= 1) {
            return true;
        }

        for (size_t i = 1; i < face_indices.size(); ++i) {
            int step = 0;
            if (reverse) {
                step = (face_indices[i - 1] - face_indices[i]) % 8;
            } else {
                step = (face_indices[i] - face_indices[i - 1]) % 8;
            }
            if (step < 0) {
                step += 8;
            }
            if (step != 1) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool center_estimates_are_compatible(
        const LdmMeshCandidate& lhs, const LdmMeshCandidate& rhs) const noexcept {
        if (!lhs.solved || !rhs.solved) {
            return true;
        }

        const Eigen::Vector3d lhs_center = lhs.pose.camera.translation();
        const Eigen::Vector3d rhs_center = rhs.pose.camera.translation();
        const double min_depth           = std::max(1.0, std::min(lhs_center.z(), rhs_center.z()));
        const double max_center_delta_m  = std::max(0.08, 0.03 * min_depth);
        return (lhs_center - rhs_center).norm() <= max_center_delta_m;
    }

    [[nodiscard]] bool
        same_assignment(const LdmMeshCandidate& lhs, const LdmMeshCandidate& rhs) const noexcept {
        return lhs.cluster_id == rhs.cluster_id && lhs.reverse_assignment == rhs.reverse_assignment
            && lhs.pair_indices == rhs.pair_indices
            && lhs.octagon_face_indices == rhs.octagon_face_indices;
    }

    [[nodiscard]] std::optional<LdmMeshCandidate> try_merge_candidates(
        const LdmMeshCandidate& base, const LdmMeshCandidate& extra,
        const std::vector<LightPair>& pairs, const OdomCameraTransform& T_odom_camera) const {
        if (!has_face_assignment(base) || !has_face_assignment(extra)) {
            return std::nullopt;
        }
        if (base.cluster_id >= 0 && extra.cluster_id >= 0 && base.cluster_id != extra.cluster_id) {
            return std::nullopt;
        }
        if (base.reverse_assignment != extra.reverse_assignment) {
            return std::nullopt;
        }
        if (!center_estimates_are_compatible(base, extra)) {
            return std::nullopt;
        }

        std::vector<int> pair_to_face(pairs.size(), -1);
        std::array<int, 8> face_to_pair{};
        face_to_pair.fill(-1);

        const auto insert_assignment = [&](const LdmMeshCandidate& candidate) -> bool {
            for (size_t i = 0; i < candidate.pair_indices.size(); ++i) {
                const int pair_idx = candidate.pair_indices[i];
                const int face_idx = candidate.octagon_face_indices[i];
                if (pair_idx < 0 || static_cast<size_t>(pair_idx) >= pairs.size()) {
                    return false;
                }
                if (face_idx < 0 || face_idx >= 8) {
                    return false;
                }

                if (pair_to_face[static_cast<size_t>(pair_idx)] != -1) {
                    if (pair_to_face[static_cast<size_t>(pair_idx)] != face_idx) {
                        return false;
                    }
                    continue;
                }
                if (face_to_pair[static_cast<size_t>(face_idx)] != -1) {
                    return false;
                }

                pair_to_face[static_cast<size_t>(pair_idx)] = face_idx;
                face_to_pair[static_cast<size_t>(face_idx)] = pair_idx;
            }
            return true;
        };

        if (!insert_assignment(base) || !insert_assignment(extra)) {
            return std::nullopt;
        }

        LdmMeshCandidate merged;
        merged.cluster_id         = base.cluster_id;
        merged.reverse_assignment = base.reverse_assignment;
        merged.pair_indices.reserve(pairs.size());
        merged.octagon_face_indices.reserve(pairs.size());
        for (size_t pair_idx = 0; pair_idx < pair_to_face.size(); ++pair_idx) {
            if (pair_to_face[pair_idx] < 0) {
                continue;
            }
            merged.pair_indices.push_back(static_cast<int>(pair_idx));
            merged.octagon_face_indices.push_back(pair_to_face[pair_idx]);
        }
        if (merged.pair_indices.size()
            <= std::max(base.pair_indices.size(), extra.pair_indices.size())) {
            return std::nullopt;
        }
        if (!face_sequence_is_consistent(merged.octagon_face_indices, merged.reverse_assignment)) {
            return std::nullopt;
        }

        const float lhs_weight = static_cast<float>(base.pair_indices.size());
        const float rhs_weight = static_cast<float>(extra.pair_indices.size());
        merged.preliminary_score =
            ((lhs_weight * base.preliminary_score) + (rhs_weight * extra.preliminary_score))
            / std::max(1.0f, lhs_weight + rhs_weight);
        evaluate_candidate(merged, pairs, T_odom_camera);
        if (!merged.solved) {
            return std::nullopt;
        }
        return merged;
    }

    [[nodiscard]] std::vector<LdmMeshCandidate> build_combined_candidates(
        const std::vector<LdmMeshCandidate>& evaluated_candidates,
        const std::vector<LightPair>& pairs, const OdomCameraTransform& T_odom_camera) const {
        std::vector<size_t> source_indices;
        source_indices.reserve(evaluated_candidates.size());
        for (size_t i = 0; i < evaluated_candidates.size(); ++i) {
            const auto& candidate = evaluated_candidates[i];
            if (!candidate.solved || candidate.pair_indices.size() < 2
                || !has_face_assignment(candidate)) {
                continue;
            }
            source_indices.push_back(i);
        }
        if (source_indices.size() < 2) {
            return {};
        }

        std::sort(source_indices.begin(), source_indices.end(), [&](size_t lhs, size_t rhs) {
            const auto& a = evaluated_candidates[lhs];
            const auto& b = evaluated_candidates[rhs];
            if (a.pair_indices.size() != b.pair_indices.size()) {
                return a.pair_indices.size() > b.pair_indices.size();
            }
            return a.score > b.score;
        });
        if (source_indices.size() > 32) {
            source_indices.resize(32);
        }

        std::vector<LdmMeshCandidate> combined_candidates;
        combined_candidates.reserve(source_indices.size());
        for (const size_t seed_idx : source_indices) {
            LdmMeshCandidate current = evaluated_candidates[seed_idx];
            bool grew                = false;

            while (true) {
                std::optional<LdmMeshCandidate> best_merge;
                for (const size_t other_idx : source_indices) {
                    if (other_idx == seed_idx) {
                        continue;
                    }

                    auto merged = try_merge_candidates(
                        current, evaluated_candidates[other_idx], pairs, T_odom_camera);
                    if (!merged.has_value()) {
                        continue;
                    }
                    if (!best_merge.has_value()
                        || merged->pair_indices.size() > best_merge->pair_indices.size()
                        || (merged->pair_indices.size() == best_merge->pair_indices.size()
                            && merged->score > best_merge->score)) {
                        best_merge = std::move(merged);
                    }
                }

                if (!best_merge.has_value()) {
                    break;
                }
                current = std::move(*best_merge);
                grew    = true;
            }

            if (!grew) {
                continue;
            }

            bool replaced_existing = false;
            for (auto& existing : combined_candidates) {
                if (!same_assignment(existing, current)) {
                    continue;
                }
                if (current.score > existing.score) {
                    existing = current;
                }
                replaced_existing = true;
                break;
            }
            if (!replaced_existing) {
                combined_candidates.push_back(std::move(current));
            }
        }

        std::sort(
            combined_candidates.begin(), combined_candidates.end(),
            [](const LdmMeshCandidate& a, const LdmMeshCandidate& b) {
                if (a.pair_indices.size() != b.pair_indices.size()) {
                    return a.pair_indices.size() > b.pair_indices.size();
                }
                return a.score > b.score;
            });
        return combined_candidates;
    }

    [[nodiscard]] cv::Point2f estimate_candidate_center_image(
        const std::vector<LightPair>& pairs, const std::vector<int>& pair_indices) const {
        cv::Point2f center(0.0f, 0.0f);
        if (pair_indices.empty()) {
            return center;
        }
        for (const int pair_idx : pair_indices) {
            center += pairs[static_cast<size_t>(pair_idx)].midpoint_px;
        }
        center *= (1.0f / static_cast<float>(pair_indices.size()));
        return center;
    }

    [[nodiscard]] std::optional<Eigen::Vector2d>
        normalized_image_point(const cv::Point2f& point_px) const {
        std::vector<cv::Point2f> undistorted;
        cv::undistortPoints(
            std::vector<cv::Point2f>{point_px}, undistorted, camera_matrix_, dist_coeffs_);
        if (undistorted.size() != 1u) {
            return std::nullopt;
        }
        return Eigen::Vector2d(
            static_cast<double>(undistorted.front().x), static_cast<double>(undistorted.front().y));
    }

    [[nodiscard]] Eigen::Vector3d bearing_for_image_point(const cv::Point2f& point_px) const {
        const auto normalized_point = normalized_image_point(point_px);
        if (!normalized_point.has_value()) {
            return Eigen::Vector3d(0.0, 0.0, 1.0);
        }

        Eigen::Vector3d bearing(normalized_point->x(), normalized_point->y(), 1.0);
        if (bearing.norm() <= 1e-9) {
            return Eigen::Vector3d(0.0, 0.0, 1.0);
        }
        return bearing.normalized();
    }

    [[nodiscard]] static Eigen::Matrix3d yaw_rotation_matrix(double yaw) {
        return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()).toRotationMatrix();
    }

    [[nodiscard]] static double normalize_yaw(double yaw) {
        return std::remainder(yaw, 2.0 * std::numbers::pi_v<double>);
    }

    [[nodiscard]] std::optional<CandidateCorrespondences> build_candidate_correspondences(
        const LdmMeshCandidate& candidate, const std::vector<LightPair>& pairs) const {
        if (!has_face_assignment(candidate)) {
            return std::nullopt;
        }

        CandidateCorrespondences correspondences;
        correspondences.obj_points_cv.reserve(candidate.pair_indices.size() * 2u);
        correspondences.img_points_cv.reserve(candidate.pair_indices.size() * 2u);
        correspondences.obj_points.reserve(candidate.pair_indices.size() * 2u);
        correspondences.normalized_points.reserve(candidate.pair_indices.size() * 2u);

        for (size_t i = 0; i < candidate.pair_indices.size(); ++i) {
            const int pair_idx = candidate.pair_indices[i];
            const int face_idx = candidate.octagon_face_indices[i];
            if (pair_idx < 0 || static_cast<size_t>(pair_idx) >= pairs.size()) {
                return std::nullopt;
            }
            if (face_idx < 0 || face_idx >= 8) {
                return std::nullopt;
            }

            const auto model_pair_points = pair_model_points_for_face(config_.geometry, face_idx);
            const std::array<cv::Point2f, 2> image_points{
                pairs[static_cast<size_t>(pair_idx)].top_center_px,
                pairs[static_cast<size_t>(pair_idx)].bottom_center_px,
            };

            for (size_t point_idx = 0; point_idx < 2u; ++point_idx) {
                const auto normalized_point = normalized_image_point(image_points[point_idx]);
                if (!normalized_point.has_value()) {
                    return std::nullopt;
                }

                const auto& model_point = model_pair_points[point_idx];
                correspondences.obj_points_cv.push_back(model_point);
                correspondences.img_points_cv.push_back(image_points[point_idx]);
                correspondences.obj_points.emplace_back(
                    static_cast<double>(model_point.x), static_cast<double>(model_point.y),
                    static_cast<double>(model_point.z));
                correspondences.normalized_points.push_back(*normalized_point);
            }
        }

        if (correspondences.obj_points.size() < 2u) {
            return std::nullopt;
        }
        return correspondences;
    }

    [[nodiscard]] static std::optional<Eigen::Vector3d>
        solve_translation_for_yaw(const CandidateCorrespondences& correspondences, double yaw) {
        const Eigen::Matrix3d rotation = yaw_rotation_matrix(yaw);
        Eigen::MatrixXd lhs(static_cast<Eigen::Index>(correspondences.obj_points.size() * 2u), 3);
        Eigen::VectorXd rhs(static_cast<Eigen::Index>(correspondences.obj_points.size() * 2u));

        Eigen::Index row = 0;
        for (size_t i = 0; i < correspondences.obj_points.size(); ++i) {
            const Eigen::Vector3d rotated = rotation * correspondences.obj_points[i];
            const double u                = correspondences.normalized_points[i].x();
            const double v                = correspondences.normalized_points[i].y();

            lhs(row, 0) = 1.0;
            lhs(row, 1) = 0.0;
            lhs(row, 2) = -u;
            rhs(row)    = u * rotated.z() - rotated.x();
            ++row;

            lhs(row, 0) = 0.0;
            lhs(row, 1) = 1.0;
            lhs(row, 2) = -v;
            rhs(row)    = v * rotated.z() - rotated.y();
            ++row;
        }

        const Eigen::Vector3d translation = lhs.colPivHouseholderQr().solve(rhs);
        if (!std::isfinite(translation.x()) || !std::isfinite(translation.y())
            || !std::isfinite(translation.z()) || translation.z() <= 1e-3) {
            return std::nullopt;
        }
        return translation;
    }

    [[nodiscard]] static double normalized_reprojection_rmse(
        const CandidateCorrespondences& correspondences, double yaw,
        const Eigen::Vector3d& translation) {
        const Eigen::Matrix3d rotation = yaw_rotation_matrix(yaw);
        double sq_error_sum            = 0.0;
        size_t count                   = 0;

        for (size_t i = 0; i < correspondences.obj_points.size(); ++i) {
            const Eigen::Vector3d cam_point =
                rotation * correspondences.obj_points[i] + translation;
            if (!std::isfinite(cam_point.x()) || !std::isfinite(cam_point.y())
                || !std::isfinite(cam_point.z()) || cam_point.z() <= 1e-6) {
                return std::numeric_limits<double>::infinity();
            }
            const Eigen::Vector2d projected(
                cam_point.x() / cam_point.z(), cam_point.y() / cam_point.z());
            const Eigen::Vector2d delta = projected - correspondences.normalized_points[i];
            sq_error_sum += delta.squaredNorm();
            ++count;
        }

        if (count == 0u) {
            return std::numeric_limits<double>::infinity();
        }
        return std::sqrt(sq_error_sum / static_cast<double>(count));
    }

    [[nodiscard]] std::vector<PoseSeed>
        make_pose_seeds(const CandidateCorrespondences& correspondences) const {
        std::vector<PoseSeed> seeds;
        constexpr int kYawSeedCount = 64;
        seeds.reserve(kYawSeedCount);

        for (int i = 0; i < kYawSeedCount; ++i) {
            const double yaw = -std::numbers::pi_v<double>
                             + (2.0 * std::numbers::pi_v<double>
                                * static_cast<double>(i) / static_cast<double>(kYawSeedCount));
            auto translation = solve_translation_for_yaw(correspondences, yaw);
            if (!translation.has_value()) {
                continue;
            }
            PoseSeed seed;
            seed.yaw             = yaw;
            seed.translation     = *translation;
            seed.normalized_rmse = normalized_reprojection_rmse(correspondences, yaw, *translation);
            if (std::isfinite(seed.normalized_rmse)) {
                seeds.push_back(seed);
            }
        }

        std::sort(seeds.begin(), seeds.end(), [](const PoseSeed& lhs, const PoseSeed& rhs) {
            return lhs.normalized_rmse < rhs.normalized_rmse;
        });
        if (seeds.size() > 6u) {
            seeds.resize(6u);
        }
        return seeds;
    }

    [[nodiscard]] std::optional<PoseSeed> refine_pose_seed(
        const CandidateCorrespondences& correspondences, const PoseSeed& seed) const {
        double yaw            = seed.yaw;
        double translation[3] = {seed.translation.x(), seed.translation.y(), seed.translation.z()};
        ceres::Problem problem;
        for (size_t i = 0; i < correspondences.obj_points.size(); ++i) {
            problem.AddResidualBlock(
                NormalizedReprojectionError::Create(
                    correspondences.obj_points[i], correspondences.normalized_points[i]),
                new ceres::HuberLoss(0.005), &yaw, translation);
        }
        problem.SetParameterLowerBound(translation, 2, 1e-3);

        ceres::Solver::Options options;
        options.linear_solver_type           = ceres::DENSE_QR;
        options.max_num_iterations           = 35;
        options.minimizer_progress_to_stdout = false;
        options.function_tolerance           = 1e-12;
        options.parameter_tolerance          = 1e-12;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        PoseSeed refined;
        refined.yaw         = normalize_yaw(yaw);
        refined.translation = Eigen::Vector3d(translation[0], translation[1], translation[2]);
        if (!summary.IsSolutionUsable() || !std::isfinite(refined.yaw)
            || !std::isfinite(refined.translation.x()) || !std::isfinite(refined.translation.y())
            || !std::isfinite(refined.translation.z()) || refined.translation.z() <= 1e-3) {
            return std::nullopt;
        }

        refined.normalized_rmse =
            normalized_reprojection_rmse(correspondences, refined.yaw, refined.translation);
        if (!std::isfinite(refined.normalized_rmse)) {
            return std::nullopt;
        }
        return refined;
    }

    [[nodiscard]] std::optional<PoseSeed>
        solve_constrained_pose_seed(const CandidateCorrespondences& correspondences) const {
        const auto seeds = make_pose_seeds(correspondences);
        if (seeds.empty()) {
            return std::nullopt;
        }

        std::optional<PoseSeed> best_seed;
        for (const auto& seed : seeds) {
            const auto refined = refine_pose_seed(correspondences, seed);
            if (!refined.has_value()) {
                continue;
            }
            if (!best_seed.has_value() || refined->normalized_rmse < best_seed->normalized_rmse) {
                best_seed = *refined;
            }
        }
        return best_seed;
    }

    [[nodiscard]] float pixel_reprojection_rmse(
        const CandidateCorrespondences& correspondences, const cv::Mat& rvec,
        const cv::Mat& tvec) const {
        std::vector<cv::Point2f> reproj_points;
        cv::projectPoints(
            correspondences.obj_points_cv, rvec, tvec, camera_matrix_, dist_coeffs_, reproj_points);
        if (reproj_points.size() != correspondences.img_points_cv.size()) {
            return std::numeric_limits<float>::infinity();
        }

        double sq_error_sum = 0.0;
        for (size_t i = 0; i < reproj_points.size(); ++i) {
            const cv::Point2f delta = reproj_points[i] - correspondences.img_points_cv[i];
            sq_error_sum += static_cast<double>(delta.x) * static_cast<double>(delta.x)
                          + static_cast<double>(delta.y) * static_cast<double>(delta.y);
        }
        return static_cast<float>(std::sqrt(
            sq_error_sum / static_cast<double>(std::max<size_t>(1, reproj_points.size()))));
    }

    [[nodiscard]] std::optional<ConstrainedPoseSolution> build_pose_solution(
        const LdmMeshCandidate& candidate, const CandidateCorrespondences& correspondences,
        const Eigen::Matrix3d& rotation_cam, const Eigen::Vector3d& center_cam) const {
        if (!std::isfinite(center_cam.x()) || !std::isfinite(center_cam.y())
            || !std::isfinite(center_cam.z()) || center_cam.z() <= 1e-3) {
            return std::nullopt;
        }

        ConstrainedPoseSolution solution;
        solution.rotation_cam = rotation_cam;
        solution.center_cam   = center_cam;

        cv::Mat rotation_cv;
        cv::eigen2cv(solution.rotation_cam, rotation_cv);
        cv::Rodrigues(rotation_cv, solution.rvec);
        solution.tvec =
            (cv::Mat_<double>(3, 1) << solution.center_cam.x(), solution.center_cam.y(),
             solution.center_cam.z());

        solution.reprojection_rmse_px =
            pixel_reprojection_rmse(correspondences, solution.rvec, solution.tvec);
        if (!std::isfinite(solution.reprojection_rmse_px)) {
            return std::nullopt;
        }

        std::vector<cv::Point2f> projected_center;
        cv::projectPoints(
            std::vector<cv::Point3f>{cv::Point3f(0.0f, 0.0f, 0.0f)}, solution.rvec, solution.tvec,
            camera_matrix_, dist_coeffs_, projected_center);
        if (!projected_center.empty()) {
            solution.projected_center_image = projected_center.front();
        }

        const auto outline_model_points = volume_outline_points(config_.geometry);
        cv::projectPoints(
            std::vector<cv::Point3f>(outline_model_points.begin(), outline_model_points.end()),
            solution.rvec, solution.tvec, camera_matrix_, dist_coeffs_,
            solution.projected_outline_image);

        solution.vertical_axis_term   = projected_vertical_axis_score(solution.rvec, solution.tvec);
        solution.face_visibility_term = visible_face_score(candidate, rotation_cam, center_cam);
        return solution;
    }

    [[nodiscard]] std::optional<ConstrainedPoseSolution> solve_full_pnp_candidate_pose(
        const LdmMeshCandidate& candidate, const CandidateCorrespondences& correspondences) const {
        if (correspondences.obj_points_cv.size() < 4u) {
            return std::nullopt;
        }

        const std::array<int, 4> pnp_flags = {
            cv::SOLVEPNP_ITERATIVE,
            cv::SOLVEPNP_SQPNP,
            cv::SOLVEPNP_EPNP,
            cv::SOLVEPNP_IPPE,
        };

        std::optional<ConstrainedPoseSolution> best_solution;
        for (const int flag : pnp_flags) {
            cv::Mat rvec;
            cv::Mat tvec;
            bool solved = false;
            try {
                solved = cv::solvePnP(
                    correspondences.obj_points_cv, correspondences.img_points_cv, camera_matrix_,
                    dist_coeffs_, rvec, tvec, false, flag);
            } catch (const cv::Exception&) {
                continue;
            }
            if (!solved) {
                continue;
            }

            cv::Mat rotation_cv;
            try {
                cv::Rodrigues(rvec, rotation_cv);
            } catch (const cv::Exception&) {
                continue;
            }

            cv::Mat rotation_cv64;
            cv::Mat tvec64;
            rotation_cv.convertTo(rotation_cv64, CV_64F);
            tvec.convertTo(tvec64, CV_64F);
            if (rotation_cv64.rows != 3 || rotation_cv64.cols != 3 || tvec64.total() != 3u) {
                continue;
            }

            Eigen::Matrix3d rotation_cam;
            cv::cv2eigen(rotation_cv64, rotation_cam);
            const cv::Mat tvec_col = tvec64.reshape(1, 3);
            const Eigen::Vector3d center_cam(
                tvec_col.at<double>(0, 0), tvec_col.at<double>(1, 0), tvec_col.at<double>(2, 0));

            auto solution =
                build_pose_solution(candidate, correspondences, rotation_cam, center_cam);
            if (!solution.has_value()) {
                continue;
            }
            if (!best_solution.has_value()
                || solution->reprojection_rmse_px < best_solution->reprojection_rmse_px) {
                best_solution = std::move(solution);
            }
        }

        return best_solution;
    }

    [[nodiscard]] std::optional<ConstrainedPoseSolution> solve_single_face_candidate_pose(
        const LdmMeshCandidate& candidate, const std::vector<LightPair>& pairs) const {
        if (!has_face_assignment(candidate) || candidate.pair_indices.size() != 1u) {
            return std::nullopt;
        }

        const auto correspondences = build_candidate_correspondences(candidate, pairs);
        if (!correspondences.has_value() || correspondences->normalized_points.size() != 2u) {
            return std::nullopt;
        }

        const int face_idx = candidate.octagon_face_indices.front();
        if (face_idx < 0 || face_idx >= 8) {
            return std::nullopt;
        }

        const Eigen::Vector2d top_norm    = correspondences->normalized_points[0];
        const Eigen::Vector2d bottom_norm = correspondences->normalized_points[1];
        const double normalized_height    = bottom_norm.y() - top_norm.y();
        const double model_height         = config_.geometry.pair_center_separation_m;
        if (!std::isfinite(normalized_height) || normalized_height <= 1e-6
            || model_height <= 1e-6) {
            return std::nullopt;
        }

        const double face_center_z = model_height / normalized_height;
        const double face_center_x = 0.5 * (top_norm.x() + bottom_norm.x()) * face_center_z;
        const double face_center_y = 0.5 * (top_norm.y() + bottom_norm.y()) * face_center_z;
        const Eigen::Vector3d face_center_cam(face_center_x, face_center_y, face_center_z);

        const double yaw = static_cast<double>(face_idx) * (std::numbers::pi_v<double> / 4.0);
        const Eigen::Matrix3d rotation_cam = yaw_rotation_matrix(yaw);

        const auto model_pair = pair_model_points_for_face(config_.geometry, face_idx);
        const Eigen::Vector3d face_center_model(
            0.5 * static_cast<double>(model_pair[0].x + model_pair[1].x),
            0.5 * static_cast<double>(model_pair[0].y + model_pair[1].y),
            0.5 * static_cast<double>(model_pair[0].z + model_pair[1].z));
        const Eigen::Vector3d center_cam = face_center_cam - rotation_cam * face_center_model;

        return build_pose_solution(candidate, *correspondences, rotation_cam, center_cam);
    }

    [[nodiscard]] std::optional<ConstrainedPoseSolution> solve_constrained_candidate_pose(
        const LdmMeshCandidate& candidate, const std::vector<LightPair>& pairs) const {
        if (candidate.pair_indices.size() == 1u) {
            return solve_single_face_candidate_pose(candidate, pairs);
        }

        const auto correspondences = build_candidate_correspondences(candidate, pairs);
        if (!correspondences.has_value()) {
            return std::nullopt;
        }
        if (correspondences->obj_points.size() < 4u) {
            return std::nullopt;
        }

        std::optional<ConstrainedPoseSolution> best_solution =
            solve_full_pnp_candidate_pose(candidate, *correspondences);

        const auto seed = solve_constrained_pose_seed(*correspondences);
        if (seed.has_value()) {
            auto yaw_only_solution = build_pose_solution(
                candidate, *correspondences, yaw_rotation_matrix(seed->yaw), seed->translation);
            if (yaw_only_solution.has_value()
                && (!best_solution.has_value()
                    || yaw_only_solution->reprojection_rmse_px
                           < best_solution->reprojection_rmse_px)) {
                best_solution = std::move(yaw_only_solution);
            }
        }

        return best_solution;
    }

    [[nodiscard]] float
        projected_vertical_axis_score(const cv::Mat& rvec, const cv::Mat& tvec) const {
        const float half_axis =
            static_cast<float>(std::max(config_.geometry.pair_center_separation_m, 1e-3) * 0.5);
        std::vector<cv::Point2f> projected_axis;
        cv::projectPoints(
            std::vector<cv::Point3f>{
                cv::Point3f(0.0f, -half_axis, 0.0f), cv::Point3f(0.0f, half_axis, 0.0f)},
            rvec, tvec, camera_matrix_, dist_coeffs_, projected_axis);
        if (projected_axis.size() != 2u) {
            return 0.0f;
        }

        const cv::Point2f delta = projected_axis[1] - projected_axis[0];
        const float norm        = std::sqrt(delta.x * delta.x + delta.y * delta.y);
        if (norm <= 1e-3f) {
            return 0.0f;
        }
        return std::clamp(delta.y / norm, 0.0f, 1.0f);
    }

    [[nodiscard]] float visible_face_score(
        const LdmMeshCandidate& candidate, const Eigen::Matrix3d& rotation_cam,
        const Eigen::Vector3d& center_cam) const {
        if (!has_face_assignment(candidate)) {
            return 0.5f;
        }

        const double scale = std::max(1e-6, octagon_face_center_radius_m(config_.geometry));
        float score_sum    = 0.0f;
        int score_count    = 0;
        for (const int face_idx : candidate.octagon_face_indices) {
            const auto model_pair = pair_model_points_for_face(config_.geometry, face_idx);
            const Eigen::Vector3d face_center_model(
                0.5 * static_cast<double>(model_pair[0].x + model_pair[1].x),
                0.5 * static_cast<double>(model_pair[0].y + model_pair[1].y),
                0.5 * static_cast<double>(model_pair[0].z + model_pair[1].z));
            const Eigen::Vector3d face_center_cam = rotation_cam * face_center_model + center_cam;
            const double front_delta              = center_cam.z() - face_center_cam.z();
            score_sum +=
                std::clamp(static_cast<float>(0.5 + 0.5 * (front_delta / scale)), 0.0f, 1.0f);
            ++score_count;
        }

        if (score_count <= 0) {
            return 0.5f;
        }
        return score_sum / static_cast<float>(score_count);
    }

    void evaluate_candidate(
        LdmMeshCandidate& candidate, const std::vector<LightPair>& pairs,
        const OdomCameraTransform& T_odom_camera) const {
        candidate.estimated_center_image_px =
            estimate_candidate_center_image(pairs, candidate.pair_indices);
        if (!face_sequence_is_consistent(
                candidate.octagon_face_indices, candidate.reverse_assignment)) {
            candidate.solved               = false;
            candidate.depth_valid          = false;
            candidate.reprojection_rmse_px = std::numeric_limits<float>::infinity();
            candidate.score                = 0.35f * candidate.preliminary_score;
            return;
        }

        const auto solution = solve_constrained_candidate_pose(candidate, pairs);
        if (!solution.has_value()) {
            candidate.solved               = false;
            candidate.depth_valid          = false;
            candidate.reprojection_rmse_px = std::numeric_limits<float>::infinity();
            candidate.score                = 0.35f * candidate.preliminary_score;
            return;
        }

        candidate.solved                    = true;
        candidate.reprojection_rmse_px      = solution->reprojection_rmse_px;
        candidate.estimated_center_image_px = solution->projected_center_image;

        const auto transform_cam =
            CameraLdmTransform::from_rt(solution->rotation_cam, solution->center_cam);
        candidate.pose = LdmCandidatePose{
            .camera = transform_cam,
            .odom   = T_odom_camera * transform_cam,
        };
        candidate.projected_outline_image = solution->projected_outline_image;

        const float vertical_axis_term   = solution->vertical_axis_term;
        const float face_visibility_term = solution->face_visibility_term;

        const double rmse_limit = (candidate.pair_indices.size() >= 3)
                                    ? config_.rmse_stable_threshold_px
                                    : config_.rmse_constrained_threshold_px;
        candidate.depth_valid =
            std::isfinite(solution->center_cam.x()) && std::isfinite(solution->center_cam.y())
            && std::isfinite(solution->center_cam.z()) && solution->center_cam.z() > 1e-3
            && candidate.reprojection_rmse_px <= static_cast<float>(rmse_limit);

        const float pair_term = pair_count_score(static_cast<int>(candidate.pair_indices.size()));
        const float rmse_term = std::exp(
            -candidate.reprojection_rmse_px
            / static_cast<float>(std::max(1.0, config_.rmse_constrained_threshold_px)));
        const float depth_bonus = candidate.depth_valid ? 0.22f : -0.10f;
        candidate.score         = std::clamp(
            0.32f * candidate.preliminary_score + 0.20f * pair_term + 0.18f * rmse_term
                + 0.15f * vertical_axis_term + 0.15f * face_visibility_term + depth_bonus,
            0.0f, 1.2f);
    }

    [[nodiscard]] float pair_count_score(int pair_count) const {
        switch (pair_count) {
        case 1: return 0.25f;
        case 2: return 0.55f;
        case 3: return 0.82f;
        default: return 0.96f;
        }
    }

    [[nodiscard]] float confidence_for(const LdmMeshCandidate& candidate) const {
        const float base_pair = pair_count_score(static_cast<int>(candidate.pair_indices.size()));
        const float prelim    = candidate.preliminary_score;
        const float quality_term = (candidate.pair_indices.size() >= 3u) ? 0.95f : 0.70f;

        const float rmse_term =
            std::isfinite(candidate.reprojection_rmse_px)
                ? std::exp(
                      -candidate.reprojection_rmse_px
                      / static_cast<float>(std::max(1.0, config_.rmse_constrained_threshold_px)))
                : 0.0f;
        return std::clamp(
            0.35f * base_pair + 0.25f * prelim + 0.20f * rmse_term + 0.20f * quality_term, 0.0f,
            1.0f);
    }

private:
    LdmDetectorConfig config_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
};

} // namespace fcs::L2::ldm
