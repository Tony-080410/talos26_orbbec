#ifndef RUNE_DETECTOR_TYPES_HPP_
#define RUNE_DETECTOR_TYPES_HPP_

// 3rd party
#include <opencv2/core.hpp>
#include <vector>

namespace fyt::rune {
struct rune {};
enum class Status {
    SUCCESS,
    ARROW_FAILURE,
    ARMOR_FAILURE,
    CENTER_FAILURE,
    FAILURE
}; // 成功，箭头检测失败，装甲板检测失败，中心R检测失败

constexpr double RUNE_PAN_REAL_DIS = 0.15; // 靶的宽度
constexpr double RUNE_R2PANCENTER  = 0.7;  // 中心到靶的距离

struct Points {
    cv::Point2f center;
    std::vector<cv::Point2f> corners;
};

struct KeyPoint {
    cv::Point2f lu = cv::Point2f(0, 0);    // 左上
    cv::Point2f ru = cv::Point2f(0, 0);    // 右上
    cv::Point2f rd = cv::Point2f(0, 0);    // 右下
    cv::Point2f ld = cv::Point2f(0, 0);    // 左下
};

struct Light {
    Light() = default;
    Light(
        const std::vector<cv::Point>& cnt, const cv::Rect2f& globalRoi, const cv::Rect2f& localRoi,
        const cv::Mat& image);

    double roundness;
    std::vector<cv::Point> contour;
    cv::Point2f center;
    cv::Size2f size;
    float angle;
    double contourArea;
    double area;
    cv::RotatedRect rotated;
    double ratio;
};

struct CenterR {
    CenterR() = default;
    void set(const Light& light);
    Light light;
    cv::Point2f center;
    cv::Rect boundingRect;
};

struct Target {
    cv::Point2f center = cv::Point2f(0, 0);
    double roll        = 0;
    std::vector<cv::Point> contour;
    void set(const Light& l);
    KeyPoint keypnt;
    bool initKey;
    std::vector<cv::Point3f> small_points3d = {
        {0.0f,                      0.0f,                                        0.0f}, // P0
        {0.0f,  RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f}, // P1
        {0.0f,  RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f}, // P2
        {0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f}, // P3
        {0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f}, // P4
        {0.0f,                      0.0f,                            RUNE_R2PANCENTER}  // P5
    };

    std::vector<cv::Point3f> points3d = {
        {-0.18f,                      0.0f,                                        0.0f}, // P0
        {  0.0f,  RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f}, // P1
        {  0.0f,  RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f}, // P2
        {  0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f}, // P3
        {  0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f}, // P4
        {  0.0f,                      0.0f,                            RUNE_R2PANCENTER}  // P5
    };
};

struct Arrow {
    Arrow() = default;
    bool
        set(const std::vector<Light>& points, const cv::Point2f& roi, const cv::Mat& image,
            bool debug_mode = false);
    std::vector<cv::Point> contour;
    cv::Point2f center;
    cv::Size2f size;
    double angle;
    double area;
    cv::RotatedRect rotated;
    double ratio;
    double fillratio;
};

} // namespace fyt::rune
#endif // RUNE_DETECTOR_TYPES_HPP_
