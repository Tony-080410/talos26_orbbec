#pragma once

#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

namespace auto_aim {

    /**
     * @brief Ground truth armor annotation
     *
     * Label format: color type label x1 y1 x2 y2 x3 y3 x4 y4
     * - color: 0=Blue, 1=Red, 2=Gray, 3=Purple
     * - type: 0=Small, 1=Large
     * - label: 0-7 (ArmorLabel enum)
     * - x1,y1...x4,y4: normalized corner points (0-1 range)
     */
    struct GroundTruth {
        uint8_t color{}; // 0=Blue, 1=Red, 2=Gray, 3=Purple
        uint8_t type{}; // 0=Small, 1=Large
        uint8_t label{}; // ArmorLabel enum
        std::array<cv::Point2f, 4> points; // Corner points (pixel coords)
        cv::Point2f center; // Computed center

        // Compute bounding box
        [[nodiscard]] constexpr cv::Rect2f bounding_box() const {
            const float min_x = std::min({points[0].x, points[1].x, points[2].x, points[3].x});
            const float min_y = std::min({points[0].y, points[1].y, points[2].y, points[3].y});
            const float max_x = std::max({points[0].x, points[1].x, points[2].x, points[3].x});
            const float max_y = std::max({points[0].y, points[1].y, points[2].y, points[3].y});
            return {min_x, min_y, max_x - min_x, max_y - min_y};
        }
    };

    /**
     * @brief Single dataset frame with image and annotations
     */
    struct DatasetFrame {
        std::string name;
        cv::Mat image;
        std::vector<GroundTruth> ground_truths;
        uint64_t frame_id{};
    };

    /**
     * @brief Dataset loader for armor detection benchmark
     *
     * Supports two label formats:
     * 1. Simple: class x1 y1 x2 y2 x3 y3 x4 y4
     * 2. Extended: color type label x1 y1 x2 y2 x3 y3 x4 y4
     */
    class DatasetLoader {
    public:
        explicit DatasetLoader(const std::string &dataset_path) :
            dataset_path_(dataset_path), image_dir_(dataset_path + "/images") {
            // Try both "label" and "labels" folder names
            label_dir_ = std::filesystem::path(dataset_path + "/labels");
            if (!std::filesystem::exists(label_dir_)) {
                label_dir_ = std::filesystem::path(dataset_path + "/label");
            }
            scan_frames();
        }

        [[nodiscard]] size_t size() const { return frame_names_.size(); }
        [[nodiscard]] bool empty() const { return frame_names_.empty(); }

        /**
         * @brief Load a specific frame by index
         */
        [[nodiscard]] std::optional<DatasetFrame> load_frame(const size_t index) const {
            if (index >= frame_names_.size()) {
                return std::nullopt;
            }

            const auto &name = frame_names_[index];
            DatasetFrame frame;
            frame.name = name;
            frame.frame_id = index;

            // Load image
            const auto image_path = image_dir_ / (name + ".jpg");
            frame.image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (frame.image.empty()) {
                return std::nullopt;
            }

            // Load labels
            if (const auto label_path = label_dir_ / (name + ".txt"); !load_labels(label_path, frame)) {
                return std::nullopt;
            }

            return frame;
        }

        /**
         * @brief Iterator-style loading
         */
        class Iterator {
        public:
            Iterator(const DatasetLoader *loader, const size_t index) : loader_(loader), index_(index) {}

            std::optional<DatasetFrame> operator*() const { return loader_->load_frame(index_); }

            Iterator &operator++() {
                ++index_;
                return *this;
            }
            bool operator!=(const Iterator &other) const { return index_ != other.index_; }

        private:
            const DatasetLoader *loader_;
            size_t index_;
        };

        [[nodiscard]] Iterator begin() const { return {this, 0}; }
        [[nodiscard]] Iterator end() const { return {this, frame_names_.size()}; }

    private:
        std::filesystem::path dataset_path_;
        std::filesystem::path image_dir_;
        std::filesystem::path label_dir_;
        std::vector<std::string> frame_names_;

        void scan_frames() {
            if (!std::filesystem::exists(image_dir_)) {
                return;
            }

            for (const auto &entry: std::filesystem::directory_iterator(image_dir_)) {
                if (entry.path().extension() == ".jpg") {
                    auto stem = entry.path().stem().string();
                    // Check if corresponding label exists
                    if (auto label_path = label_dir_ / (stem + ".txt"); std::filesystem::exists(label_path)) {
                        frame_names_.push_back(stem);
                    }
                }
            }

            // Sort by name for deterministic order
            std::ranges::sort(frame_names_);
        }

        static bool load_labels(const std::filesystem::path &path, DatasetFrame &frame) {
            std::ifstream file(path);
            if (!file.is_open()) {
                return false;
            }

            auto img_width = static_cast<float>(frame.image.cols);
            auto img_height = static_cast<float>(frame.image.rows);

            std::string line;
            while (std::getline(file, line)) {
                if (line.empty())
                    continue;

                std::istringstream iss(line);
                GroundTruth gt;

                // Try to parse all values into a vector first
                std::vector<float> values;
                float val;
                while (iss >> val) {
                    values.push_back(val);
                }

                float x1, y1, x2, y2, x3, y3, x4, y4;

                if (values.size() == 9) {
                    // Simple format: class x1 y1 x2 y2 x3 y3 x4 y4
                    int cls = static_cast<int>(values[0]);
                    // Class encoding: assume class contains color info
                    // Common encoding: 0-3 = blue armors, 4-7 = red armors
                    gt.color = cls >= 4 ? 1 : 0; // 0=blue, 1=red
                    gt.type = 0; // Default small
                    gt.label = static_cast<uint8_t>(cls % 4);
                    x1 = values[1];
                    y1 = values[2];
                    x2 = values[3];
                    y2 = values[4];
                    x3 = values[5];
                    y3 = values[6];
                    x4 = values[7];
                    y4 = values[8];
                } else if (values.size() == 11) {
                    // Extended format: color type label x1 y1 x2 y2 x3 y3 x4 y4
                    gt.color = static_cast<uint8_t>(values[0]);
                    gt.type = static_cast<uint8_t>(values[1]);
                    gt.label = static_cast<uint8_t>(values[2]);
                    x1 = values[3];
                    y1 = values[4];
                    x2 = values[5];
                    y2 = values[6];
                    x3 = values[7];
                    y3 = values[8];
                    x4 = values[9];
                    y4 = values[10];
                } else {
                    continue; // Unknown format
                }

                // Convert normalized coords to pixel coords
                gt.points[0] = cv::Point2f(x1 * img_width, y1 * img_height);
                gt.points[1] = cv::Point2f(x2 * img_width, y2 * img_height);
                gt.points[2] = cv::Point2f(x3 * img_width, y3 * img_height);
                gt.points[3] = cv::Point2f(x4 * img_width, y4 * img_height);

                // Compute center
                gt.center = (gt.points[0] + gt.points[1] + gt.points[2] + gt.points[3]) / 4.0f;

                frame.ground_truths.push_back(gt);
            }

            return true;
        }
    };

    /**
     * @brief Detection accuracy metrics
     */
    struct AccuracyMetrics {
        int total_gt{0}; // Total ground truth armors
        int total_det{0}; // Total detected armors
        int true_positives{0}; // Correctly detected
        int false_positives{0}; // Detected but wrong
        int false_negatives{0}; // Missed detections

        double iou_sum{0.0}; // Sum of IoU for matched pairs
        int iou_count{0}; // Number of matched pairs

        // Color accuracy
        int color_correct{0};
        int color_total{0};

        // Type accuracy
        int type_correct{0};
        int type_total{0};

        [[nodiscard]] constexpr double precision() const {
            return total_det > 0 ? static_cast<double>(true_positives) / total_det : 0.0;
        }

        [[nodiscard]] constexpr double recall() const {
            return total_gt > 0 ? static_cast<double>(true_positives) / total_gt : 0.0;
        }

        [[nodiscard]] constexpr double f1_score() const {
            const double p = precision();
            const double r = recall();
            return p + r > 0 ? 2 * p * r / (p + r) : 0.0;
        }

        [[nodiscard]] constexpr double mean_iou() const { return iou_count > 0 ? iou_sum / iou_count : 0.0; }

        [[nodiscard]] constexpr double color_accuracy() const {
            return color_total > 0 ? static_cast<double>(color_correct) / color_total : 0.0;
        }

        [[nodiscard]] double type_accuracy() const {
            return type_total > 0 ? static_cast<double>(type_correct) / type_total : 0.0;
        }

        void print() const {
            fmt::print("\n=== Detection Accuracy Report ===\n");
            fmt::print("Ground Truth: {}, Detected: {}\n", total_gt, total_det);
            fmt::print("TP: {}, FP: {}, FN: {}\n", true_positives, false_positives, false_negatives);
            fmt::print("Precision: {:.2f}%\n", precision() * 100);
            fmt::print("Recall:    {:.2f}%\n", recall() * 100);
            fmt::print("F1 Score:  {:.2f}%\n", f1_score() * 100);
            fmt::print("Mean IoU:  {:.2f}%\n", mean_iou() * 100);
            fmt::print("Color Acc: {:.2f}% ({}/{})\n", color_accuracy() * 100, color_correct, color_total);
            fmt::print("Type Acc:  {:.2f}% ({}/{})\n", type_accuracy() * 100, type_correct, type_total);
        }

        AccuracyMetrics &operator+=(const AccuracyMetrics &other) {
            total_gt += other.total_gt;
            total_det += other.total_det;
            true_positives += other.true_positives;
            false_positives += other.false_positives;
            false_negatives += other.false_negatives;
            iou_sum += other.iou_sum;
            iou_count += other.iou_count;
            color_correct += other.color_correct;
            color_total += other.color_total;
            type_correct += other.type_correct;
            type_total += other.type_total;
            return *this;
        }
    };

    /**
     * @brief Calculate IoU between two bounding boxes
     */
    constexpr double calculate_iou(const cv::Rect2f &a, const cv::Rect2f &b) {
        const float x1 = std::max(a.x, b.x);
        const float y1 = std::max(a.y, b.y);
        const float x2 = std::min(a.x + a.width, b.x + b.width);
        const float y2 = std::min(a.y + a.height, b.y + b.height);

        if (x2 <= x1 || y2 <= y1) {
            return 0.0;
        }

        const float intersection = (x2 - x1) * (y2 - y1);
        const float union_area = a.area() + b.area() - intersection;

        return union_area > 0 ? intersection / union_area : 0.0;
    }

} // namespace auto_aim
