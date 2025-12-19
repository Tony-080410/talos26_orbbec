/**
 * @file armor_ratio_test.cpp
 * @brief Unit tests for ArmorFull ratio perspective correction
 *
 * Test Strategy:
 * 1. Define a known ratio ideal rectangular armor plate (3D space)
 * 2. Apply different yaw and pitch rotations
 * 3. Project rotated 3D points to 2D screen coordinates using pinhole camera model
 * 4. Construct LightbarFull from projected points, then construct ArmorFull
 * 5. Verify ArmorFull::ratio matches the original ratio within acceptable error
 */

#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "auto_aim/detector.hpp"

namespace {

    // Armor plate physical dimensions (mm)
    constexpr double ARMOR_WIDTH = 135.0; // Distance between lightbar centers
    constexpr double LIGHTBAR_LENGTH = 55.0; // Lightbar height
    constexpr double EXPECTED_RATIO = ARMOR_WIDTH / LIGHTBAR_LENGTH; // ≈ 2.45

    // Camera intrinsic parameters (typical 1280x1024 industrial camera)
    constexpr double FX = 1000.0; // Focal length x (pixels)
    constexpr double FY = 1000.0; // Focal length y (pixels)
    constexpr double CX = 640.0; // Principal point x
    constexpr double CY = 512.0; // Principal point y
    constexpr double CAMERA_DISTANCE = 3000.0; // Distance from camera to armor center (mm)

    // Error tolerance
    constexpr double RATIO_RELATIVE_ERROR_THRESHOLD = 0.05; // 5%

    /**
     * @brief Convert degrees to radians
     */
    constexpr double deg2rad(const double degrees) { return degrees * CV_PI / 180.0; }

    /**
     * @brief 3D point structure
     */
    struct Point3D {
        double x, y, z;

        constexpr Point3D(const double x_, const double y_, const double z_) : x(x_), y(y_), z(z_) {}

        constexpr Point3D rotate_pitch(const double pitch_rad) const {
            // Rotation around X axis (pitch)
            const double cos_p = std::cos(pitch_rad);
            const double sin_p = std::sin(pitch_rad);
            return {x, y * cos_p - z * sin_p, y * sin_p + z * cos_p};
        }

        constexpr Point3D rotate_yaw(const double yaw_rad) const {
            // Rotation around Y axis (yaw)
            const double cos_y = std::cos(yaw_rad);
            const double sin_y = std::sin(yaw_rad);
            return {x * cos_y + z * sin_y, y, -x * sin_y + z * cos_y};
        }

        [[nodiscard]] constexpr cv::Point2f project() const {
            // Pinhole camera model: u = fx * X/Z + cx, v = fy * Y/Z + cy
            if (z <= 0) {
                // Point behind camera
                return {-1, -1};
            }
            auto u = static_cast<float>(FX * x / z + CX);
            auto v = static_cast<float>(FY * y / z + CY);
            return {u, v};
        }
    };

    /**
     * @brief Create armor plate 3D corner points
     *
     * Armor plate is centered at origin, facing +Z direction
     * Returns: {left_top, right_top, right_bottom, left_bottom}
     *
     * Coordinate system:
     * - X: right
     * - Y: down
     * - Z: forward (camera looking direction)
     */
    constexpr std::array<Point3D, 4> create_armor_3d(const double width, const double height) {
        const double half_w = width / 2.0;
        const double half_h = height / 2.0;

        return {
                Point3D(-half_w, -half_h, 0), // left_top
                Point3D(+half_w, -half_h, 0), // right_top
                Point3D(+half_w, +half_h, 0), // right_bottom
                Point3D(-half_w, +half_h, 0), // left_bottom
        };
    }

    /**
     * @brief Apply yaw and pitch rotation to armor, then translate to camera distance
     */
    constexpr std::array<Point3D, 4> transform_armor(const std::array<Point3D, 4> &armor, const double yaw_deg,
                                                     const double pitch_deg, const double distance = CAMERA_DISTANCE) {
        const double yaw_rad = deg2rad(yaw_deg);
        const double pitch_rad = deg2rad(pitch_deg);

        std::array<Point3D, 4> result = armor;

        for (auto &pt: result) {
            // First rotate around Y (yaw), then around X (pitch)
            pt = pt.rotate_yaw(yaw_rad);
            pt = pt.rotate_pitch(pitch_rad);
            // Translate to camera distance
            pt.z += distance;
        }

        return result;
    }

    /**
     * @brief Project 3D armor to 2D and return corner points
     */
    constexpr std::array<cv::Point2f, 4> project_armor(const std::array<Point3D, 4> &armor_3d) {
        return {
                armor_3d[0].project(), // left_top
                armor_3d[1].project(), // right_top
                armor_3d[2].project(), // right_bottom
                armor_3d[3].project(), // left_bottom
        };
    }

    /**
     * @brief Create LightbarFull from top and bottom points
     *
     * Simulates what the detector would extract from an image
     */
    constexpr talos::LightbarFull create_lightbar_from_points(const cv::Point2f &top, const cv::Point2f &bottom,
                                                              const std::size_t id) {
        talos::LightbarFull lb;
        lb.id = id;
        lb.top = top;
        lb.bottom = bottom;
        lb.center = (top + bottom) / 2.0f;
        lb.top2bottom = bottom - top;
        lb.length = cv::norm(lb.top2bottom);
        lb.angle = std::atan2(lb.top2bottom.y, lb.top2bottom.x);
        lb.angle_error = std::abs(lb.angle - CV_PI / 2.0);

        // Create a minimal rotated rect for compatibility
        std::vector pts = {top, bottom};
        const cv::Point2f diff = bottom - top;
        constexpr float width = 5.0f; // Approximate lightbar width
        cv::Point2f perp(-diff.y, diff.x);
        perp = perp / cv::norm(perp) * (width / 2.0f);

        const std::vector rect_pts = {cv::Point(top + perp), cv::Point(top - perp), cv::Point(bottom - perp),
                                      cv::Point(bottom + perp)};
        lb.rotated_rect = cv::minAreaRect(rect_pts);
        lb.width = width;
        lb.ratio = lb.length / lb.width;

        return lb;
    }

    /**
     * @brief Create ArmorFull from projected 2D points and compute its ratio
     */
    constexpr talos::ArmorFull create_armor_from_projection(const std::array<cv::Point2f, 4> &pts_2d) {
        // pts_2d: {left_top, right_top, right_bottom, left_bottom}
        const talos::LightbarFull left_lb = create_lightbar_from_points(pts_2d[0], pts_2d[3], 0);
        const talos::LightbarFull right_lb = create_lightbar_from_points(pts_2d[1], pts_2d[2], 1);

        // Use the ArmorFull constructor which computes ratio
        talos::ArmorFull armor(left_lb, right_lb);
        return armor;
    }

    /**
     * @brief Calculate the relative error between measured and expected ratio
     */
    constexpr double calculate_relative_error(const double measured, const double expected) {
        return std::abs(measured - expected) / expected;
    }

    /**
     * @brief Test helper: run a single perspective test case
     */
    constexpr void run_perspective_test(const double yaw_deg, const double pitch_deg, const double expected_ratio,
                                        const double error_threshold = RATIO_RELATIVE_ERROR_THRESHOLD) {
        // Create 3D armor plate
        constexpr auto armor_3d = create_armor_3d(ARMOR_WIDTH, LIGHTBAR_LENGTH);

        // Transform with rotation and translation
        const auto transformed = transform_armor(armor_3d, yaw_deg, pitch_deg);

        // Project to 2D
        const auto pts_2d = project_armor(transformed);

        // Verify all points are valid (in front of camera)
        for (const auto &pt: pts_2d) {
            ASSERT_GE(pt.x, 0) << "Point behind camera at yaw=" << yaw_deg << ", pitch=" << pitch_deg;
            ASSERT_GE(pt.y, 0) << "Point behind camera at yaw=" << yaw_deg << ", pitch=" << pitch_deg;
        }

        // Create ArmorFull from projected points
        const talos::ArmorFull armor = create_armor_from_projection(pts_2d);

        // Calculate relative error
        const double relative_error = calculate_relative_error(armor.ratio, expected_ratio);

        // Log for debugging
        SCOPED_TRACE(testing::Message() << "yaw=" << yaw_deg << "°, pitch=" << pitch_deg << "°"
                                        << ", measured_ratio=" << armor.ratio << ", expected_ratio=" << expected_ratio
                                        << ", relative_error=" << (relative_error * 100) << "%");

        EXPECT_LE(relative_error, error_threshold)
                << "Ratio error exceeds " << (error_threshold * 100) << "% threshold";
    }

} // namespace

// ============================================================================
// Test Cases: Perspective Distortion Correction
// ============================================================================

class ArmorRatioPerspectiveTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No setup needed
    }
};

/**
 * @brief Test case: yaw=0°, pitch=0° (directly facing camera)
 *
 * This is the baseline case where no perspective correction is needed.
 * The measured ratio should match the physical ratio exactly.
 */
TEST_F(ArmorRatioPerspectiveTest, FacingCamera) {
    run_perspective_test(0.0, 0.0, EXPECTED_RATIO, 0.01); // Very tight tolerance
}

/**
 * @brief Test case: yaw=30°, pitch=0° (pure yaw rotation)
 *
 * Yaw affects the apparent width of the armor plate.
 * Without correction: apparent_width = actual_width * cos(yaw)
 * The correction algorithm should recover the original ratio.
 */
TEST_F(ArmorRatioPerspectiveTest, PureYaw30) { run_perspective_test(30.0, 0.0, EXPECTED_RATIO); }

/**
 * @brief Test case: yaw=0°, pitch=20° (pure pitch rotation)
 *
 * Pitch affects the apparent length of the lightbars.
 * Without correction: apparent_height = actual_height * cos(pitch)
 * The correction algorithm should recover the original ratio.
 */
TEST_F(ArmorRatioPerspectiveTest, PurePitch20) { run_perspective_test(0.0, 20.0, EXPECTED_RATIO); }

/**
 * @brief Test case: yaw=30°, pitch=20° (combined rotation)
 *
 * Both yaw and pitch affect the observed geometry.
 * The correction must handle both distortions simultaneously.
 */
TEST_F(ArmorRatioPerspectiveTest, CombinedYaw30Pitch20) { run_perspective_test(30.0, 20.0, EXPECTED_RATIO); }

/**
 * @brief Test case: yaw=-30°, pitch=-15° (reversed combined rotation)
 *
 * Test negative angles to ensure the algorithm handles
 * rotations in both directions correctly.
 */
TEST_F(ArmorRatioPerspectiveTest, ReverseCombinedYawNeg30PitchNeg15) {
    run_perspective_test(-30.0, -15.0, EXPECTED_RATIO);
}

/**
 * @brief Test case: yaw=45°, pitch=30° (large angle combined rotation)
 *
 * Tests the algorithm at more extreme viewing angles.
 * This is near the limit of practical detection scenarios.
 */
TEST_F(ArmorRatioPerspectiveTest, LargeAngleYaw45Pitch30) {
    // Allow slightly larger error for extreme angles
    run_perspective_test(45.0, 30.0, EXPECTED_RATIO, 0.08);
}

// ============================================================================
// Test Cases: Edge Cases and Symmetry
// ============================================================================

/**
 * @brief Test symmetry: opposite yaw angles should produce same ratio error
 */
TEST_F(ArmorRatioPerspectiveTest, YawSymmetry) {
    constexpr auto armor_3d = create_armor_3d(ARMOR_WIDTH, LIGHTBAR_LENGTH);

    const auto transformed_pos = transform_armor(armor_3d, 25.0, 0.0);
    const auto transformed_neg = transform_armor(armor_3d, -25.0, 0.0);

    const auto pts_pos = project_armor(transformed_pos);
    const auto pts_neg = project_armor(transformed_neg);

    const auto armor_pos = create_armor_from_projection(pts_pos);
    const auto armor_neg = create_armor_from_projection(pts_neg);

    // Ratios should be identical due to symmetry
    EXPECT_NEAR(armor_pos.ratio, armor_neg.ratio, 0.01)
            << "Yaw symmetry violated: +25° ratio=" << armor_pos.ratio << ", -25° ratio=" << armor_neg.ratio;
}

/**
 * @brief Test symmetry: opposite pitch angles should produce same ratio error
 */
TEST_F(ArmorRatioPerspectiveTest, PitchSymmetry) {
    constexpr auto armor_3d = create_armor_3d(ARMOR_WIDTH, LIGHTBAR_LENGTH);

    const auto transformed_pos = transform_armor(armor_3d, 0.0, 20.0);
    const auto transformed_neg = transform_armor(armor_3d, 0.0, -20.0);

    const auto pts_pos = project_armor(transformed_pos);
    const auto pts_neg = project_armor(transformed_neg);

    const auto armor_pos = create_armor_from_projection(pts_pos);
    const auto armor_neg = create_armor_from_projection(pts_neg);

    // Ratios should be identical due to symmetry
    EXPECT_NEAR(armor_pos.ratio, armor_neg.ratio, 0.01)
            << "Pitch symmetry violated: +20° ratio=" << armor_pos.ratio << ", -20° ratio=" << armor_neg.ratio;
}

/**
 * @brief Test: Ratio should stay approximately constant with yaw (corrected)
 *
 * After perspective correction, the ratio should stay close to the true value
 * regardless of yaw angle. Small variations are acceptable.
 */
TEST_F(ArmorRatioPerspectiveTest, RatioStaysConstantWithYaw_Corrected) {
    constexpr auto armor_3d = create_armor_3d(ARMOR_WIDTH, LIGHTBAR_LENGTH);

    std::vector<std::pair<double, double>> yaw_ratios;
    for (double yaw = 0; yaw <= 45; yaw += 15) {
        auto transformed = transform_armor(armor_3d, yaw, 0.0);
        auto pts_2d = project_armor(transformed);
        auto armor = create_armor_from_projection(pts_2d);
        yaw_ratios.emplace_back(yaw, armor.ratio);
    }

    // Verify ratio stays approximately constant (within 5% of expected)
    for (const auto &[yaw, ratio]: yaw_ratios) {
        double relative_error = calculate_relative_error(ratio, EXPECTED_RATIO);
        EXPECT_LE(relative_error, RATIO_RELATIVE_ERROR_THRESHOLD)
                << "Corrected ratio at yaw=" << yaw << "° is " << ratio << ", expected ~" << EXPECTED_RATIO;
    }
}

/**
 * @brief Test: Ratio should stay approximately constant with pitch (corrected)
 *
 * After perspective correction, the ratio should stay close to the true value
 * regardless of pitch angle. Small variations are acceptable.
 */
TEST_F(ArmorRatioPerspectiveTest, RatioStaysConstantWithPitch_Corrected) {
    constexpr auto armor_3d = create_armor_3d(ARMOR_WIDTH, LIGHTBAR_LENGTH);

    std::vector<std::pair<double, double>> pitch_ratios;
    for (double pitch = 0; pitch <= 30; pitch += 10) {
        auto transformed = transform_armor(armor_3d, 0.0, pitch);
        auto pts_2d = project_armor(transformed);
        auto armor = create_armor_from_projection(pts_2d);
        pitch_ratios.emplace_back(pitch, armor.ratio);
    }

    // Verify ratio stays approximately constant (within 5% of expected)
    for (const auto &[pitch, ratio]: pitch_ratios) {
        double relative_error = calculate_relative_error(ratio, EXPECTED_RATIO);
        EXPECT_LE(relative_error, RATIO_RELATIVE_ERROR_THRESHOLD)
                << "Corrected ratio at pitch=" << pitch << "° is " << ratio << ", expected ~" << EXPECTED_RATIO;
    }
}

// ============================================================================
// Test Cases: Big Armor Plate
// ============================================================================

constexpr double BIG_ARMOR_WIDTH = 230.0;
constexpr double BIG_LIGHTBAR_LENGTH = 55.0;
constexpr double BIG_EXPECTED_RATIO = BIG_ARMOR_WIDTH / BIG_LIGHTBAR_LENGTH; // ≈ 4.18

TEST_F(ArmorRatioPerspectiveTest, BigArmor_FacingCamera) {
    constexpr auto armor_3d = create_armor_3d(BIG_ARMOR_WIDTH, BIG_LIGHTBAR_LENGTH);
    const auto transformed = transform_armor(armor_3d, 0.0, 0.0);
    const auto pts_2d = project_armor(transformed);
    const auto armor = create_armor_from_projection(pts_2d);

    const double relative_error = calculate_relative_error(armor.ratio, BIG_EXPECTED_RATIO);
    EXPECT_LE(relative_error, 0.01);
}

TEST_F(ArmorRatioPerspectiveTest, BigArmor_CombinedRotation) {
    constexpr auto armor_3d = create_armor_3d(BIG_ARMOR_WIDTH, BIG_LIGHTBAR_LENGTH);
    const auto transformed = transform_armor(armor_3d, 25.0, 15.0);
    const auto pts_2d = project_armor(transformed);
    const auto armor = create_armor_from_projection(pts_2d);

    const double relative_error = calculate_relative_error(armor.ratio, BIG_EXPECTED_RATIO);
    EXPECT_LE(relative_error, RATIO_RELATIVE_ERROR_THRESHOLD);
}

// ============================================================================
// Test Cases: Different Distances
// ============================================================================

TEST_F(ArmorRatioPerspectiveTest, NearDistance) {
    constexpr auto armor_3d = create_armor_3d(ARMOR_WIDTH, LIGHTBAR_LENGTH);
    const auto transformed = transform_armor(armor_3d, 20.0, 10.0, 1500.0); // 1.5m
    const auto pts_2d = project_armor(transformed);
    const auto armor = create_armor_from_projection(pts_2d);

    const double relative_error = calculate_relative_error(armor.ratio, EXPECTED_RATIO);
    EXPECT_LE(relative_error, RATIO_RELATIVE_ERROR_THRESHOLD);
}

TEST_F(ArmorRatioPerspectiveTest, FarDistance) {
    constexpr auto armor_3d = create_armor_3d(ARMOR_WIDTH, LIGHTBAR_LENGTH);
    const auto transformed = transform_armor(armor_3d, 20.0, 10.0, 6000.0); // 6m
    const auto pts_2d = project_armor(transformed);
    const auto armor = create_armor_from_projection(pts_2d);

    const double relative_error = calculate_relative_error(armor.ratio, EXPECTED_RATIO);
    EXPECT_LE(relative_error, RATIO_RELATIVE_ERROR_THRESHOLD);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
