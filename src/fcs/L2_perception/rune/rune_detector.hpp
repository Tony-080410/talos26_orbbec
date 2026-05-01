
#ifndef RUNE_DETECTOR_RUNE_DETECTOR_HPP_
#define RUNE_DETECTOR_RUNE_DETECTOR_HPP_

// std
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "L2_perception/rune/rune_config.hpp"
#include "core/armor_types.hpp"
#include "frame.hpp"
#include "matrix.hpp"
#include <expected>
#include <string>

#include "L2_perception/rune/rune_detector_types.hpp"

namespace fyt::rune {

class RuneDetector {
public:
    explicit RuneDetector(const fcs::rune::RuneDetectorConfig& params = {});

    bool detect(const cv::Mat& frame, fcs::ArmorColor color);

    std::expected<fast_tf::TransformMatrixd<rune, fast_tf::odom>, std::string> solve(
        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double timestamp_sec,
        const fast_tf::TransformMatrixd<fast_tf::odom, fast_tf::camera_optical>& odom_T_cam);

    void setKeyPoints();
    void setLocalRoi();
    void setGlobalRoi();

    cv::Mat localMask;
    cv::Rect2f globalRoi;
    std::vector<cv::Rect2f> targetROIs;
    cv::Rect2f centerRoi;
    cv::Mat arrowImg;
    cv::Mat targetImg;
    cv::Mat rCenterImg;
    CenterR rcenter;
    std::vector<Target> targets;
    std::vector<Arrow> arrows;
    Status status;
    double timestamp_sec{0.0};

private:
    std::expected<cv::Mat, std::string> processPictures(const cv::Mat& src, fcs::ArmorColor color);
    bool isArrowRoiMatched(const Arrow& arrow, const cv::Rect2f& roi, const cv::Rect2f& globalRoi);

    bool detectCenterR(fcs::ArmorColor color);
    bool detectAllArrows(fcs::ArmorColor color);
    bool detectAllTargets(fcs::ArmorColor color);

    int image_width_;
    int image_height_;
    fcs::rune::RuneDetectorConfig params_;

    // 上一帧追踪的靶信息（用于目标连续性）
    cv::Point2f last_tracked_target_center_{0, 0};
    double last_tracked_target_roll_{0.0};
    double last_timestamp_ = 0;
    bool has_last_target_{false};

public:
    // 重投影调试信息
    std::vector<cv::Point2f> debug_detected_points_;
    std::vector<cv::Point2f> debug_reprojected_points_;
    double debug_reproj_error_{0.0};
};
} // namespace fyt::rune
#endif // RUNE_DETECTOR_RUNE_DETECTOR_HPP_
