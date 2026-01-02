#include "fcs/calibration/charuco_detector.hpp"
#include "fcs/calibration/chessboard_detector.hpp"

#ifdef HAVE_OPENCV_ARUCO
# include <opencv2/imgproc.hpp>
# include <opencv2/objdetect/aruco_detector.hpp>
# include <opencv2/objdetect/charuco_detector.hpp>
#endif

namespace fcs::calibration {

#ifdef HAVE_OPENCV_ARUCO

CharucoDetector::CharucoDetector(
    uint32_t width, uint32_t height, double square_size, double marker_size,
    ArucoDictionary dictionary) noexcept
    : board_size_(static_cast<int>(width), static_cast<int>(height))
    , square_size_(square_size)
    , marker_size_(marker_size)
    , dictionary_(cv::aruco::getPredefinedDictionary(to_opencv_dict(dictionary)))
    , charuco_board_(
          cv::Size(static_cast<int>(width), static_cast<int>(height)),
          static_cast<float>(square_size), static_cast<float>(marker_size), dictionary_)
    , detector_params_()
    , object_points_() {
    // Get charuco corner object points
    // ChArUco corners are at the intersections of squares
    const int corners_width  = static_cast<int>(width) - 1;
    const int corners_height = static_cast<int>(height) - 1;

    object_points_.reserve(static_cast<size_t>(corners_width) * corners_height);
    for (int i = 0; i < corners_height; ++i) {
        for (int j = 0; j < corners_width; ++j) {
            object_points_.emplace_back(
                static_cast<float>((j + 1) * square_size),
                static_cast<float>((i + 1) * square_size), 0.0f);
        }
    }
}

std::expected<CornerDetection, std::string>
    CharucoDetector::detect(const cv::Mat& image, timestamp_ns_t ts) noexcept {
    CornerDetection result;
    result.timestamp_ns = ts;
    result.image        = image.clone();

    // Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Create detector
    cv::aruco::CharucoDetector detector(
        charuco_board_, cv::aruco::CharucoParameters(), cv::aruco::DetectorParameters());

    // Detect ChArUco corners
    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int> marker_ids;

    detector.detectBoard(gray, charuco_corners, charuco_ids, marker_corners, marker_ids);

    if (charuco_corners.empty() || charuco_corners.size() < 6) {
        result.success = false;
        return std::unexpected("Not enough ChArUco corners detected");
    }

    // Map detected corners to object points
    result.image_points = std::move(charuco_corners);
    result.object_points.clear();
    result.object_points.reserve(charuco_ids.size());

    const int corners_width = board_size_.width - 1;
    for (int id : charuco_ids) {
        int row = id / corners_width;
        int col = id % corners_width;
        result.object_points.emplace_back(
            static_cast<float>((col + 1) * square_size_),
            static_cast<float>((row + 1) * square_size_), 0.0f);
    }

    result.success = true;
    return result;
}

const std::vector<cv::Point3f>& CharucoDetector::object_points() const noexcept {
    return object_points_;
}

cv::Mat CharucoDetector::draw_corners(
    const cv::Mat& image, const CornerDetection& detection) const noexcept {
    cv::Mat output = image.clone();
    if (output.channels() == 1) {
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    }

    if (detection.success && !detection.image_points.empty()) {
        // Draw charuco corners
        for (size_t i = 0; i < detection.image_points.size(); ++i) {
            cv::circle(output, detection.image_points[i], 5, cv::Scalar(0, 255, 0), 2);
            cv::putText(
                output, std::to_string(i), detection.image_points[i] + cv::Point2f(5, -5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
        }
    }

    return output;
}

#endif // HAVE_OPENCV_ARUCO

// Factory function implementation
std::expected<std::unique_ptr<CalibrationBoard>, std::string>
    create_board(const BoardConfig& board_config, const CharucoConfig& charuco_config) noexcept {
    switch (board_config.type) {
    case BoardType::Chessboard:
        return std::make_unique<ChessboardDetector>(
            board_config.width, board_config.height, board_config.square_size);

    case BoardType::ChArUco:
#ifdef HAVE_OPENCV_ARUCO
        return std::make_unique<CharucoDetector>(
            board_config.width, board_config.height, board_config.square_size,
            charuco_config.marker_size, charuco_config.dictionary);
#else
        return std::unexpected("ChArUco support not available (OpenCV aruco module not found)");
#endif

    case BoardType::CirclesGrid: return std::unexpected("CirclesGrid not yet implemented");
    }

    return std::unexpected("Unknown board type");
}

} // namespace fcs::calibration
