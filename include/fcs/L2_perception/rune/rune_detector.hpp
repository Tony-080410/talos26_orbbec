
#ifndef RUNE_DETECTOR_RUNE_DETECTOR_HPP_
#define RUNE_DETECTOR_RUNE_DETECTOR_HPP_

// std
#include <Eigen/Dense>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

#include <expected>
#include <string>

#include "fcs/L2_perception/rune/rune_detector_types.hpp"

namespace fyt::rune {

class RuneDetector {
public:
    RuneDetector(
        EnemyColor detect_color, int arrow_threshold, int target_threshold, int rcenter_threshold);

    bool detect(const cv::Mat& frame, int image_width, int image_height);

    std::expected<Eigen::Matrix4d, std::string> solve(
        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double timestamp_sec,
        const Eigen::Matrix4d& odom_T_cam);

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
    cv::Mat processPictures(const cv::Mat& src);
    bool isArrowRoiMatched(const Arrow& arrow, const cv::Rect2f& roi, const cv::Rect2f& globalRoi);

    bool detectCenterR();
    bool detectAllArrows();
    bool detectAllTargets();

    EnemyColor detect_color_;
    int arrow_threshold_;
    int target_threshold_;
    int rcenter_threshold_;

    int image_width_;
    int image_height_;

    // 上一帧追踪的靶位置（用于目标连续性）
    cv::Point2f last_tracked_target_center_{0, 0};
    double last_timestamp_ = 0;
    bool has_last_target_{false};
};
} // namespace fyt::rune
#endif // RUNE_DETECTOR_RUNE_DETECTOR_HPP_
