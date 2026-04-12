#include "L2_perception/ldm/ldm_detector.hpp"

#include "L2_perception/ldm/ldm_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace fcs::L2::ldm {

namespace {

[[nodiscard]] float
    min_candidate_score(const LdmMeshCandidate& candidate, const LdmDetectorConfig& config) {
    return static_cast<float>(
        (candidate.pair_indices.size() <= 2) ? config.min_preliminary_candidate_score_two_pair
                                             : config.min_preliminary_candidate_score);
}

[[nodiscard]] bool candidate_passes_detection_gate(
    const LdmMeshCandidate& candidate, const std::vector<LightPair>& pairs,
    const LdmDetectorConfig& config) {
    if (static_cast<int>(candidate.pair_indices.size()) < config.min_pairs_for_detection) {
        return false;
    }
    if (candidate.preliminary_score < min_candidate_score(candidate, config)) {
        return false;
    }

    if (candidate.pair_indices.size() <= 2) {
        double mean_center_dy_px = 0.0;
        int valid_pair_count     = 0;
        for (const int pair_idx : candidate.pair_indices) {
            if (pair_idx < 0 || static_cast<size_t>(pair_idx) >= pairs.size()) {
                continue;
            }
            mean_center_dy_px += pairs[static_cast<size_t>(pair_idx)].center_dy_px;
            ++valid_pair_count;
        }
        if (valid_pair_count == 0) {
            return false;
        }
        mean_center_dy_px /= static_cast<double>(valid_pair_count);
        if (mean_center_dy_px < config.min_two_pair_mean_center_dy_px) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] cv::Point2f rect_center(const cv::Rect& rect) {
    return cv::Point2f(
        rect.x + static_cast<float>(rect.width) * 0.5f,
        rect.y + static_cast<float>(rect.height) * 0.5f);
}

[[nodiscard]] cv::Rect2f rect2f_from_rect(const cv::Rect& rect) {
    return cv::Rect2f(
        static_cast<float>(rect.x), static_cast<float>(rect.y), static_cast<float>(rect.width),
        static_cast<float>(rect.height));
}

[[nodiscard]] cv::Mat threshold_target_color(const cv::Mat& image_bgr, ArmorColor color) {
    cv::Mat hsv;
    cv::cvtColor(image_bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    switch (color) {
    case ArmorColor::Blue:
        cv::inRange(hsv, cv::Scalar(90, 70, 70), cv::Scalar(140, 255, 255), mask);
        break;
    case ArmorColor::Red: {
        cv::Mat mask1, mask2;
        cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(10, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(160, 80, 80), cv::Scalar(180, 255, 255), mask2);
        mask = mask1 | mask2;
        break;
    }
    default: {
        cv::Mat red  = threshold_target_color(image_bgr, ArmorColor::Red);
        cv::Mat blue = threshold_target_color(image_bgr, ArmorColor::Blue);
        mask         = red | blue;
        break;
    }
    }

    // Remove thin red borders / HUD overlays so they cannot connect unrelated pixels
    // into a single frame-sized contour.
    constexpr int border_px = 2;
    if (mask.rows > border_px * 2 && mask.cols > border_px * 2) {
        mask.rowRange(0, border_px).setTo(0);
        mask.rowRange(mask.rows - border_px, mask.rows).setTo(0);
        mask.colRange(0, border_px).setTo(0);
        mask.colRange(mask.cols - border_px, mask.cols).setTo(0);
    }
    return mask;
}

[[nodiscard]] std::vector<LightBlob> detect_light_blobs(
    const cv::Mat& image_bgr, const LdmDetectorConfig& config, ArmorColor color) {
    if (image_bgr.empty()) {
        return {};
    }

    cv::Mat mask = threshold_target_color(image_bgr, color);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBlob> blobs;
    blobs.reserve(contours.size());
    for (const auto& contour : contours) {
        const double contour_area = cv::contourArea(contour);
        if (contour_area < static_cast<double>(config.min_blob_area_px)) {
            continue;
        }

        const cv::Rect rect = cv::boundingRect(contour);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }

        const float aspect_ratio =
            static_cast<float>(rect.width) / static_cast<float>(std::max(rect.height, 1));
        if (aspect_ratio < static_cast<float>(config.min_blob_aspect_ratio)
            || aspect_ratio > static_cast<float>(config.max_blob_aspect_ratio)) {
            continue;
        }

        const float fill_ratio = static_cast<float>(contour_area)
                               / static_cast<float>(std::max(1, rect.width * rect.height));
        if (fill_ratio < static_cast<float>(config.min_blob_fill_ratio)) {
            continue;
        }

        blobs.push_back(
            LightBlob{
                .rect         = rect2f_from_rect(rect),
                .center_px    = rect_center(rect),
                .area_px      = static_cast<float>(contour_area),
                .aspect_ratio = aspect_ratio,
                .fill_ratio   = fill_ratio});
    }

    std::sort(blobs.begin(), blobs.end(), [](const LightBlob& a, const LightBlob& b) {
        if (a.center_px.x != b.center_px.x) {
            return a.center_px.x < b.center_px.x;
        }
        return a.center_px.y < b.center_px.y;
    });
    return blobs;
}

[[nodiscard]] float
    pair_score(const LightBlob& top, const LightBlob& bottom, const LdmDetectorConfig& config) {
    const float avg_w = (top.rect.width + bottom.rect.width) * 0.5f;
    const float avg_h = (top.rect.height + bottom.rect.height) * 0.5f;
    if (avg_w <= 1e-3f || avg_h <= 1e-3f) {
        return 0.0f;
    }

    const cv::Point2f top_center    = top.center_px;
    const cv::Point2f bottom_center = bottom.center_px;
    const float dx                  = std::abs(bottom_center.x - top_center.x);
    const float dy                  = bottom_center.y - top_center.y;
    if (dy <= 0.0f) {
        return 0.0f;
    }

    const float dx_limit =
        static_cast<float>(config.max_pair_center_dx_ratio) * std::max(avg_w, 1.0f);
    if (dx > dx_limit) {
        return 0.0f;
    }

    const float min_dy =
        static_cast<float>(config.min_pair_center_dy_ratio) * std::max(avg_h, 1.0f);
    const float max_dy =
        static_cast<float>(config.max_pair_center_dy_ratio) * std::max(avg_h, 1.0f);
    if (dy < min_dy || dy > max_dy) {
        return 0.0f;
    }

    const float width_delta = std::abs(top.rect.width - bottom.rect.width) / std::max(avg_w, 1.0f);
    const float height_delta =
        std::abs(top.rect.height - bottom.rect.height) / std::max(avg_h, 1.0f);
    if (width_delta > static_cast<float>(config.max_pair_size_delta_ratio)
        || height_delta > static_cast<float>(config.max_pair_size_delta_ratio)) {
        return 0.0f;
    }

    const float angle_deg = std::abs(
        std::atan2(bottom_center.x - top_center.x, bottom_center.y - top_center.y) * 180.0f
        / static_cast<float>(std::numbers::pi_v<double>));
    if (angle_deg > static_cast<float>(config.max_pair_angle_deg)) {
        return 0.0f;
    }

    const float dx_score     = std::max(0.0f, 1.0f - dx / std::max(dx_limit, 1.0f));
    const float width_score  = std::max(0.0f, 1.0f - width_delta);
    const float height_score = std::max(0.0f, 1.0f - height_delta);
    const float fill_score   = 0.5f * (top.fill_ratio + bottom.fill_ratio);
    return std::clamp(
        0.35f * dx_score + 0.20f * width_score + 0.20f * height_score + 0.25f * fill_score, 0.0f,
        1.0f);
}

struct PairCandidate {
    size_t top_idx{0};
    size_t bottom_idx{0};
    float score{0.0f};
};

[[nodiscard]] std::vector<LightPair>
    build_light_pairs(const std::vector<LightBlob>& blobs, const LdmDetectorConfig& config) {
    std::vector<LightPair> pairs;
    if (blobs.size() < 2) {
        return pairs;
    }

    std::vector<PairCandidate> pair_candidates;
    pair_candidates.reserve(blobs.size() * 2);
    std::vector<std::vector<int>> adjacency(blobs.size());

    for (size_t first_idx = 0; first_idx < blobs.size(); ++first_idx) {
        for (size_t second_idx = first_idx + 1; second_idx < blobs.size(); ++second_idx) {
            size_t top_idx    = first_idx;
            size_t bottom_idx = second_idx;
            if (blobs[top_idx].center_px.y > blobs[bottom_idx].center_px.y) {
                std::swap(top_idx, bottom_idx);
            }

            const float score = pair_score(blobs[top_idx], blobs[bottom_idx], config);
            if (score <= 0.0f) {
                continue;
            }

            const int candidate_idx = static_cast<int>(pair_candidates.size());
            pair_candidates.push_back(
                PairCandidate{
                    .top_idx    = top_idx,
                    .bottom_idx = bottom_idx,
                    .score      = score,
                });
            adjacency[top_idx].push_back(candidate_idx);
            adjacency[bottom_idx].push_back(candidate_idx);
        }
    }

    if (pair_candidates.empty()) {
        return pairs;
    }

    std::vector<int> selected_candidate_indices;
    selected_candidate_indices.reserve(blobs.size() / 2);

    if (blobs.size() <= 20) {
        const uint32_t full_mask = (1u << blobs.size()) - 1u;
        std::vector<float> best_score(1u << blobs.size(), -1.0f);
        std::vector<int> best_choice(1u << blobs.size(), -2);

        const auto solve = [&](auto&& self, uint32_t used_mask) -> float {
            if (used_mask == full_mask) {
                return 0.0f;
            }

            float& cached_score = best_score[used_mask];
            if (cached_score >= 0.0f) {
                return cached_score;
            }

            size_t first_unused = 0;
            while ((used_mask & (1u << first_unused)) != 0u) {
                ++first_unused;
            }

            float best = self(self, used_mask | (1u << first_unused));
            int choice = -1;
            for (const int candidate_idx : adjacency[first_unused]) {
                const auto& candidate = pair_candidates[static_cast<size_t>(candidate_idx)];
                const size_t other_idx =
                    (candidate.top_idx == first_unused) ? candidate.bottom_idx : candidate.top_idx;
                const uint32_t other_bit = (1u << other_idx);
                if ((used_mask & other_bit) != 0u) {
                    continue;
                }

                const float candidate_score =
                    candidate.score + self(self, used_mask | (1u << first_unused) | other_bit);
                if (candidate_score > best) {
                    best   = candidate_score;
                    choice = candidate_idx;
                }
            }

            cached_score           = best;
            best_choice[used_mask] = choice;
            return best;
        };

        solve(solve, 0u);

        uint32_t used_mask = 0u;
        while (used_mask != full_mask) {
            size_t first_unused = 0;
            while ((used_mask & (1u << first_unused)) != 0u) {
                ++first_unused;
            }

            const int candidate_idx = best_choice[used_mask];
            if (candidate_idx < 0) {
                used_mask |= (1u << first_unused);
                continue;
            }

            const auto& candidate = pair_candidates[static_cast<size_t>(candidate_idx)];
            const size_t other_idx =
                (candidate.top_idx == first_unused) ? candidate.bottom_idx : candidate.top_idx;
            selected_candidate_indices.push_back(candidate_idx);
            used_mask |= (1u << first_unused) | (1u << other_idx);
        }
    } else {
        std::vector<int> candidate_order(pair_candidates.size());
        std::iota(candidate_order.begin(), candidate_order.end(), 0);
        std::sort(
            candidate_order.begin(), candidate_order.end(), [&](const int lhs, const int rhs) {
                const auto& a = pair_candidates[static_cast<size_t>(lhs)];
                const auto& b = pair_candidates[static_cast<size_t>(rhs)];
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                const float a_dx =
                    std::abs(blobs[a.bottom_idx].center_px.x - blobs[a.top_idx].center_px.x);
                const float b_dx =
                    std::abs(blobs[b.bottom_idx].center_px.x - blobs[b.top_idx].center_px.x);
                return a_dx < b_dx;
            });

        std::vector<bool> used(blobs.size(), false);
        for (const int candidate_idx : candidate_order) {
            const auto& candidate = pair_candidates[static_cast<size_t>(candidate_idx)];
            if (used[candidate.top_idx] || used[candidate.bottom_idx]) {
                continue;
            }
            used[candidate.top_idx]    = true;
            used[candidate.bottom_idx] = true;
            selected_candidate_indices.push_back(candidate_idx);
        }
    }

    pairs.reserve(selected_candidate_indices.size());
    for (const int candidate_idx : selected_candidate_indices) {
        const auto& candidate      = pair_candidates[static_cast<size_t>(candidate_idx)];
        const auto& top            = blobs[candidate.top_idx];
        const auto& bottom         = blobs[candidate.bottom_idx];
        const cv::Point2f midpoint = (top.center_px + bottom.center_px) * 0.5f;
        pairs.push_back(
            LightPair{
                .top_blob_index    = static_cast<int>(candidate.top_idx),
                .bottom_blob_index = static_cast<int>(candidate.bottom_idx),
                .top_center_px     = top.center_px,
                .bottom_center_px  = bottom.center_px,
                .midpoint_px       = midpoint,
                .center_dx_px      = std::abs(bottom.center_px.x - top.center_px.x),
                .center_dy_px      = bottom.center_px.y - top.center_px.y,
                .score             = candidate.score});
    }

    std::sort(pairs.begin(), pairs.end(), [](const LightPair& a, const LightPair& b) {
        return a.midpoint_px.x < b.midpoint_px.x;
    });
    return pairs;
}

[[nodiscard]] std::optional<LdmDetection>
    detect_laser_module(const cv::Mat& image, const LdmDetectorConfig& config, ArmorColor color) {
    auto blobs = detect_light_blobs(image, config, color);
    auto pairs = build_light_pairs(blobs, config);
    if (pairs.empty()) {
        return std::nullopt;
    }

    auto mesh_candidates = build_preliminary_mesh_candidates(pairs, config);
    mesh_candidates.erase(
        std::remove_if(
            mesh_candidates.begin(), mesh_candidates.end(),
            [&](const LdmMeshCandidate& candidate) {
                return !candidate_passes_detection_gate(candidate, pairs, config);
            }),
        mesh_candidates.end());
    if (mesh_candidates.empty()) {
        return std::nullopt;
    }

    std::vector<int> selected_pair_indices;
    selected_pair_indices.reserve(mesh_candidates.size() * 4);
    for (const auto& candidate : mesh_candidates) {
        selected_pair_indices.insert(
            selected_pair_indices.end(), candidate.pair_indices.begin(),
            candidate.pair_indices.end());
    }
    std::sort(selected_pair_indices.begin(), selected_pair_indices.end());
    selected_pair_indices.erase(
        std::unique(selected_pair_indices.begin(), selected_pair_indices.end()),
        selected_pair_indices.end());
    if (selected_pair_indices.empty()) {
        return std::nullopt;
    }

    std::vector<LightPair> selected_pairs;
    selected_pairs.reserve(selected_pair_indices.size());
    std::vector<int> selected_blob_indices;
    selected_blob_indices.reserve(selected_pair_indices.size() * 2);
    std::vector<int> pair_remap(pairs.size(), -1);
    for (const int pair_idx : selected_pair_indices) {
        if (pair_idx < 0 || static_cast<size_t>(pair_idx) >= pairs.size()) {
            continue;
        }
        const auto& pair                          = pairs[static_cast<size_t>(pair_idx)];
        pair_remap[static_cast<size_t>(pair_idx)] = static_cast<int>(selected_pairs.size());
        selected_pairs.push_back(pair);
        selected_blob_indices.push_back(pair.top_blob_index);
        selected_blob_indices.push_back(pair.bottom_blob_index);
    }
    if (selected_pairs.empty()) {
        return std::nullopt;
    }

    std::sort(selected_blob_indices.begin(), selected_blob_indices.end());
    selected_blob_indices.erase(
        std::unique(selected_blob_indices.begin(), selected_blob_indices.end()),
        selected_blob_indices.end());

    std::vector<LightBlob> selected_blobs;
    selected_blobs.reserve(selected_blob_indices.size());
    std::vector<int> blob_remap(blobs.size(), -1);
    for (const int old_blob_idx : selected_blob_indices) {
        if (old_blob_idx < 0 || static_cast<size_t>(old_blob_idx) >= blobs.size()) {
            continue;
        }
        blob_remap[static_cast<size_t>(old_blob_idx)] = static_cast<int>(selected_blobs.size());
        selected_blobs.push_back(blobs[static_cast<size_t>(old_blob_idx)]);
    }

    for (auto& pair : selected_pairs) {
        if (pair.top_blob_index >= 0
            && static_cast<size_t>(pair.top_blob_index) < blob_remap.size()) {
            pair.top_blob_index = blob_remap[static_cast<size_t>(pair.top_blob_index)];
        }
        if (pair.bottom_blob_index >= 0
            && static_cast<size_t>(pair.bottom_blob_index) < blob_remap.size()) {
            pair.bottom_blob_index = blob_remap[static_cast<size_t>(pair.bottom_blob_index)];
        }
    }

    for (auto& candidate : mesh_candidates) {
        for (auto& pair_idx : candidate.pair_indices) {
            if (pair_idx < 0 || static_cast<size_t>(pair_idx) >= pair_remap.size()) {
                pair_idx = -1;
                continue;
            }
            pair_idx = pair_remap[static_cast<size_t>(pair_idx)];
        }
        candidate.pair_indices.erase(
            std::remove(candidate.pair_indices.begin(), candidate.pair_indices.end(), -1),
            candidate.pair_indices.end());
    }
    mesh_candidates.erase(
        std::remove_if(
            mesh_candidates.begin(), mesh_candidates.end(),
            [](const LdmMeshCandidate& candidate) { return candidate.pair_indices.empty(); }),
        mesh_candidates.end());
    if (mesh_candidates.empty()) {
        return std::nullopt;
    }

    LdmDetection detection;
    detection.valid                  = true;
    detection.color                  = color;
    detection.blobs                  = std::move(selected_blobs);
    detection.pairs                  = std::move(selected_pairs);
    detection.mesh_candidates        = std::move(mesh_candidates);
    detection.rect                   = bounding_rect_from_pairs(detection.pairs);
    detection.selected_candidate_idx = 0;
    detection.center_image_px        = detection.mesh_candidates.front().estimated_center_image_px;
    return detection;
}

} // namespace

LdmDetector::LdmDetector(LdmDetectorConfig config) noexcept
    : config_(config)
    , target_color_(config.target_color) {}

void LdmDetector::set_target_color(ArmorColor color) noexcept { target_color_ = color; }
ArmorColor LdmDetector::get_target_color() const noexcept { return target_color_; }

std::expected<std::optional<LdmDetection>, DetectorError>
    LdmDetector::detect(const cv::Mat& image) const noexcept {
    if (image.empty()) {
        return std::unexpected(DetectorError::InvalidImage);
    }
    return detect_laser_module(image, config_, target_color_);
}
} // namespace fcs::L2::ldm
