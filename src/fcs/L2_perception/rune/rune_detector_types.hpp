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

enum class EnemyColor {
    RED   = 0,
    BLUE  = 1,
    WHITE = 2,
};

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

    std::vector<cv::Point3f> points3d = {
        {0.0f,                      0.0f,                                        0.0f}, // P0
        {0.0f,  RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f}, // P1
        {0.0f,  RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f}, // P2
        {0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER - RUNE_PAN_REAL_DIS / 2.0f}, // P3
        {0.0f, -RUNE_PAN_REAL_DIS / 2.0f, RUNE_R2PANCENTER + RUNE_PAN_REAL_DIS / 2.0f}, // P4
        {0.0f,                      0.0f,                            RUNE_R2PANCENTER}  // P5
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

// 能量机关检测参数
struct RuneDetectorParams {
    // 调试模式，启用时输出详细错误信息
    bool debug_mode{false};

    // 流水灯参数
    double min_roundness{0.6};
    double max_roundness{1.0};
    int min_arrow_light_num{3};
    int max_arrow_light_num{15};
    double min_arrow_aspect_ratio{2.0};
    double max_arrow_aspect_ratio{12.0};
    double max_arrow_area{5000.0};
    double min_arrow_light_area{10.0};
    double max_arrow_light_area{250.0};
    double min_arrow_light_aspect_ratio{1.5};
    double max_arrow_light_aspect_ratio{5.0};
    double max_same_arrow_area_ratio{5.0};

    // ROI参数
    double global_roi_length_ratio{1.5};
    double local_roi_distance_ratio{6.0};
    int local_roi_width{150};

    // R中心参数
    double max_center_aspect_ratio{3.0};

    // 靶参数
    int min_target_light_area{400};
    int max_target_light_area{30000};
    int min_target_light_contour_area{300};
    int max_target_light_contour_area{10000};
    double min_target_light_aspect_ratio{0.5};
    double max_target_light_aspect_ratio{1.5};
    float min_target_roi_area{100.0f};
};

} // namespace fyt::rune
#endif // RUNE_DETECTOR_TYPES_HPP_
