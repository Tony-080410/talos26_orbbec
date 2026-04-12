#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "L2_perception/ldm/ldm_detector.hpp"
#include "L2_perception/ldm/ldm_geometry.hpp"
#include "L2_perception/ldm/ldm_solver.hpp"
#include "camera_config.hpp"

namespace {

namespace fs = std::filesystem;

fs::path source_root() { return fs::path(TALOS_SOURCE_DIR); }
fs::path ldm_dataset_root() { return source_root() / "datasets" / "ldm"; }

std::vector<fs::path> dataset_images(const fs::path& bucket_dir) {
    std::vector<fs::path> images;
    if (!fs::exists(bucket_dir)) {
        return images;
    }

    for (const auto& entry : fs::directory_iterator(bucket_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".png") {
            continue;
        }
        images.push_back(entry.path());
    }

    std::sort(images.begin(), images.end());
    return images;
}

fcs::CameraConfig make_test_camera_config() {
    fcs::CameraConfig config;
    config.camera_matrix << 1600.0, 0.0, 1280.0, 0.0, 1600.0, 720.0, 0.0, 0.0, 1.0;
    config.distort_coefficient << 0.0, 0.0, 0.0, 0.0, 0.0;
    config.width  = 2560;
    config.height = 1440;
    return config;
}

fcs::L2::ldm::LdmDetectorConfig make_ldm_config() {
    fcs::L2::ldm::LdmDetectorConfig config;
    config.target_color = fcs::ArmorColor::Red;
    return config;
}

cv::Mat rotation_to_rvec(const Eigen::Matrix3d& rotation) {
    cv::Mat rotation_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rotation_cv.at<double>(r, c) = rotation(r, c);
        }
    }

    cv::Mat rvec;
    cv::Rodrigues(rotation_cv, rvec);
    return rvec;
}

std::vector<fcs::L2::ldm::LightPair> make_projected_pairs(
    const std::vector<int>& face_indices, const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation, const fcs::CameraConfig& camera_config,
    const fcs::L2::ldm::LdmDetectorConfig& config) {
    cv::Mat camera_matrix(3, 3, CV_64F);
    cv::Mat dist_coeffs(1, 5, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            camera_matrix.at<double>(r, c) = camera_config.camera_matrix(r, c);
        }
    }
    for (int i = 0; i < 5; ++i) {
        dist_coeffs.at<double>(0, i) = camera_config.distort_coefficient(i);
    }

    const cv::Mat rvec = rotation_to_rvec(rotation);
    const cv::Mat tvec =
        (cv::Mat_<double>(3, 1) << translation.x(), translation.y(), translation.z());

    std::vector<fcs::L2::ldm::LightPair> pairs;
    pairs.reserve(face_indices.size());
    for (const int face_index : face_indices) {
        const auto model_pair =
            fcs::L2::ldm::pair_model_points_for_face(config.geometry, face_index);
        std::vector<cv::Point2f> projected;
        cv::projectPoints(
            std::vector<cv::Point3f>{model_pair[0], model_pair[1]}, rvec, tvec, camera_matrix,
            dist_coeffs, projected);
        EXPECT_EQ(projected.size(), 2u);
        if (projected.size() != 2u) {
            continue;
        }

        cv::Point2f top    = projected[0];
        cv::Point2f bottom = projected[1];
        if (top.y > bottom.y) {
            std::swap(top, bottom);
        }

        pairs.push_back(
            fcs::L2::ldm::LightPair{
                .top_blob_index    = static_cast<int>(pairs.size() * 2),
                .bottom_blob_index = static_cast<int>(pairs.size() * 2 + 1),
                .top_center_px     = top,
                .bottom_center_px  = bottom,
                .midpoint_px       = (top + bottom) * 0.5f,
                .center_dx_px      = std::abs(bottom.x - top.x),
                .center_dy_px      = bottom.y - top.y,
                .score             = 1.0f});
    }

    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.midpoint_px.x < b.midpoint_px.x;
    });
    return pairs;
}

fcs::L2::ldm::LdmDetection make_detection_from_pairs(
    std::vector<fcs::L2::ldm::LightPair> pairs, const fcs::L2::ldm::LdmDetectorConfig& config) {
    fcs::L2::ldm::LdmDetection detection;
    detection.valid        = true;
    detection.color        = fcs::ArmorColor::Red;
    detection.timestamp_ns = 1234;
    detection.frame_id     = 7;
    detection.pairs        = std::move(pairs);
    detection.rect         = fcs::L2::ldm::bounding_rect_from_pairs(detection.pairs);
    detection.mesh_candidates =
        fcs::L2::ldm::build_preliminary_mesh_candidates(detection.pairs, config);
    return detection;
}

TEST(LdmDetector, DetectsAllThreePairDatasetImages) {
    const auto config = make_ldm_config();
    fcs::L2::ldm::LdmDetector detector(config);
    const auto images = dataset_images(ldm_dataset_root() / "3pair");

    ASSERT_FALSE(images.empty());
    for (const auto& image_path : images) {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        ASSERT_FALSE(image.empty()) << image_path;

        const auto result = detector.detect(image);
        ASSERT_TRUE(result.has_value()) << image_path;
        ASSERT_TRUE(result->has_value()) << image_path;
        EXPECT_EQ(result->value().pair_count(), 3u) << image_path;

        bool has_three_pair_candidate = false;
        for (const auto& candidate : result->value().mesh_candidates) {
            has_three_pair_candidate |= candidate.pair_indices.size() == 3u;
        }
        EXPECT_TRUE(has_three_pair_candidate) << image_path;
    }
}

TEST(LdmDetector, DetectsThreePairsWhenBottomBlobSortsBeforeTopBlob) {
    const auto config = make_ldm_config();
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::array<int, 3> top_xs    = {240, 300, 360};
    const std::array<int, 3> bottom_xs = {238, 298, 358};
    for (size_t i = 0; i < top_xs.size(); ++i) {
        cv::rectangle(
            image, cv::Rect(top_xs[i] - 10, 150, 20, 16), cv::Scalar(0, 0, 255), cv::FILLED);
        cv::rectangle(
            image, cv::Rect(bottom_xs[i] - 10, 250, 20, 16), cv::Scalar(0, 0, 255), cv::FILLED);
    }

    fcs::L2::ldm::LdmDetector detector(config);
    const auto result = detector.detect(image);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->has_value());
    EXPECT_EQ(result->value().pair_count(), 3u);
}

TEST(LdmDetector, DetectsAllTwoPairDatasetImages) {
    const auto config = make_ldm_config();
    fcs::L2::ldm::LdmDetector detector(config);
    const auto images = dataset_images(ldm_dataset_root() / "2pair");

    ASSERT_FALSE(images.empty());
    for (const auto& image_path : images) {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        ASSERT_FALSE(image.empty()) << image_path;

        const auto result = detector.detect(image);
        ASSERT_TRUE(result.has_value()) << image_path;
        ASSERT_TRUE(result->has_value()) << image_path;
        EXPECT_EQ(result->value().pair_count(), 2u) << image_path;

        bool has_two_pair_candidate = false;
        for (const auto& candidate : result->value().mesh_candidates) {
            has_two_pair_candidate |= candidate.pair_indices.size() == 2u;
        }
        EXPECT_TRUE(has_two_pair_candidate) << image_path;
    }
}

TEST(LdmDetector, DetectsAllTwoOrThreePairDatasetImages) {
    const auto config = make_ldm_config();
    fcs::L2::ldm::LdmDetector detector(config);
    const auto images = dataset_images(ldm_dataset_root() / "2-3pair");

    ASSERT_FALSE(images.empty());
    for (const auto& image_path : images) {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        ASSERT_FALSE(image.empty()) << image_path;

        const auto result = detector.detect(image);
        ASSERT_TRUE(result.has_value()) << image_path;
        ASSERT_TRUE(result->has_value()) << image_path;
        EXPECT_TRUE(result->value().pair_count() == 2u || result->value().pair_count() == 3u)
            << image_path;

        bool has_expected_candidate = false;
        for (const auto& candidate : result->value().mesh_candidates) {
            has_expected_candidate |= candidate.pair_indices.size() == 2u;
            has_expected_candidate |= candidate.pair_indices.size() == 3u;
        }
        EXPECT_TRUE(has_expected_candidate) << image_path;
    }
}

TEST(LdmDetector, RejectsLargeGapCandidateAsFullMesh) {
    const auto config = make_ldm_config();
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::array<int, 3> xs = {120, 160, 460};
    for (const int x : xs) {
        cv::rectangle(image, cv::Rect(x - 14, 120, 28, 18), cv::Scalar(0, 0, 255), cv::FILLED);
        cv::rectangle(image, cv::Rect(x - 14, 260, 28, 18), cv::Scalar(0, 0, 255), cv::FILLED);
    }

    fcs::L2::ldm::LdmDetector detector(config);
    const auto result = detector.detect(image);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->has_value());

    bool has_three_pair_candidate = false;
    for (const auto& candidate : result->value().mesh_candidates) {
        has_three_pair_candidate |= candidate.pair_indices.size() == 3;
    }
    EXPECT_FALSE(has_three_pair_candidate);
}

TEST(LdmDetector, RejectsAllNoneDatasetImages) {
    const auto config = make_ldm_config();
    fcs::L2::ldm::LdmDetector detector(config);
    const auto images = dataset_images(ldm_dataset_root() / "none");

    ASSERT_FALSE(images.empty());
    for (const auto& image_path : images) {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        ASSERT_FALSE(image.empty()) << image_path;

        const auto result = detector.detect(image);
        ASSERT_TRUE(result.has_value()) << image_path;
        EXPECT_FALSE(result->has_value()) << image_path;
    }
}

TEST(LdmSolver, SinglePairFallsBackToBearingOnly) {
    const auto camera_config = make_test_camera_config();
    const auto ldm_config    = make_ldm_config();
    fcs::L2::ldm::LdmSolver solver(camera_config, ldm_config);

    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d translation(0.08, -0.04, 3.8);
    auto detection = make_detection_from_pairs(
        make_projected_pairs({0}, rotation, translation, camera_config, ldm_config), ldm_config);

    const auto result = solver.solve(detection);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->depth_quality, fcs::L2::ldm::LdmDepthQuality::BearingOnly);
    EXPECT_FALSE(result->whole_center_cam.has_value());
}

TEST(LdmSolver, TwoPairsProduceConstrainedDepth) {
    const auto camera_config = make_test_camera_config();
    const auto ldm_config    = make_ldm_config();
    fcs::L2::ldm::LdmSolver solver(camera_config, ldm_config);

    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(-0.18, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d translation(-0.10, 0.06, 4.1);
    auto detection = make_detection_from_pairs(
        make_projected_pairs({0, 1}, rotation, translation, camera_config, ldm_config), ldm_config);

    const auto result = solver.solve(detection);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->depth_quality, fcs::L2::ldm::LdmDepthQuality::Constrained);
    ASSERT_TRUE(result->whole_center_cam.has_value());
    EXPECT_LT((result->whole_center_cam.value() - translation).norm(), 5e-2);
}

TEST(LdmSolver, CombinesMultipleMeshInfosIntoStableCenter) {
    const auto camera_config = make_test_camera_config();
    const auto ldm_config    = make_ldm_config();
    fcs::L2::ldm::LdmSolver solver(camera_config, ldm_config);

    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.18, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d translation(0.10, -0.03, 4.3);

    fcs::L2::ldm::LdmDetection detection;
    detection.valid        = true;
    detection.color        = fcs::ArmorColor::Red;
    detection.timestamp_ns = 4321;
    detection.frame_id     = 11;
    detection.pairs =
        make_projected_pairs({7, 0, 1, 2}, rotation, translation, camera_config, ldm_config);
    detection.rect            = fcs::L2::ldm::bounding_rect_from_pairs(detection.pairs);
    detection.mesh_candidates = {
        fcs::L2::ldm::LdmMeshCandidate{
                                       .pair_indices      = {0, 1},
                                       .preliminary_score = 0.93f,
                                       },
        fcs::L2::ldm::LdmMeshCandidate{
                                       .pair_indices      = {2, 3},
                                       .preliminary_score = 0.92f,
                                       },
    };

    const auto result = solver.solve(detection);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->depth_quality, fcs::L2::ldm::LdmDepthQuality::Stable);
    EXPECT_EQ(result->selected_pair_count, 4);
    ASSERT_TRUE(result->whole_center_cam.has_value());
    EXPECT_LT((result->whole_center_cam.value() - translation).norm(), 8e-2);

    bool has_combined_four_pair_candidate = false;
    for (const auto& candidate : result->mesh_candidates) {
        has_combined_four_pair_candidate |= candidate.pair_indices.size() == 4u;
    }
    EXPECT_TRUE(has_combined_four_pair_candidate);
}

TEST(LdmSolver, ThreePairsRecoverStableCenterAndOdomProjection) {
    const auto camera_config = make_test_camera_config();
    const auto ldm_config    = make_ldm_config();
    fcs::L2::ldm::LdmSolver solver(camera_config, ldm_config);

    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.22, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d translation(0.14, -0.05, 4.6);
    auto detection = make_detection_from_pairs(
        make_projected_pairs({7, 0, 1}, rotation, translation, camera_config, ldm_config),
        ldm_config);

    const auto T_camera_odom =
        fast_tf::TransformMatrixd<fast_tf::camera_optical, fast_tf::odom>::from_translation(
            1.0, 2.0, 3.0);
    const auto result = solver.solve(detection, T_camera_odom);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->depth_quality, fcs::L2::ldm::LdmDepthQuality::Stable);
    ASSERT_TRUE(result->whole_center_cam.has_value());
    ASSERT_TRUE(result->whole_center_odom.has_value());
    EXPECT_LT((result->whole_center_cam.value() - translation).norm(), 5e-2);
    EXPECT_LT(
        (result->whole_center_odom.value() - (translation + Eigen::Vector3d(1.0, 2.0, 3.0))).norm(),
        5e-2);
}

} // namespace
