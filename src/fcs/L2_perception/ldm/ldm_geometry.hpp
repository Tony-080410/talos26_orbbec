#pragma once

#include "ldm_config.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

namespace fcs::L2::ldm {

[[nodiscard]] inline double octagon_face_center_radius_m(const LdmGeometryConfig& geometry) {
    return geometry.octagon_circumradius_m * std::cos(std::numbers::pi_v<double> / 8.0);
}

[[nodiscard]] inline std::array<cv::Point3f, 2>
    pair_model_points_for_face(const LdmGeometryConfig& geometry, int face_index) {
    const double angle      = static_cast<double>(face_index) * (std::numbers::pi_v<double> / 4.0);
    const double radius     = octagon_face_center_radius_m(geometry);
    const double half_y_sep = geometry.pair_center_separation_m * 0.5;
    const float x           = static_cast<float>(radius * std::sin(angle));
    const float z           = static_cast<float>(radius * std::cos(angle));
    const float top_y       = static_cast<float>(-half_y_sep);
    const float bottom_y    = static_cast<float>(half_y_sep);
    return {cv::Point3f(x, top_y, z), cv::Point3f(x, bottom_y, z)};
}

[[nodiscard]] inline cv::Rect2f bounding_rect_from_pairs(const std::vector<LightPair>& pairs) {
    if (pairs.empty()) {
        return {};
    }

    std::vector<cv::Point2f> pts;
    pts.reserve(pairs.size() * 2);
    for (const auto& pair : pairs) {
        pts.push_back(pair.top_center_px);
        pts.push_back(pair.bottom_center_px);
    }
    return cv::boundingRect(pts);
}

[[nodiscard]] inline float pair_group_score(
    const std::vector<LightPair>& sorted_pairs, size_t start, size_t length,
    const LdmDetectorConfig& config) {
    if (length == 0) {
        return 0.0f;
    }

    const auto group_begin = sorted_pairs.begin() + static_cast<std::ptrdiff_t>(start);
    const auto group_end   = group_begin + static_cast<std::ptrdiff_t>(length);

    float mean_pair_score = 0.0f;
    float mean_pair_dy    = 0.0f;
    for (auto it = group_begin; it != group_end; ++it) {
        mean_pair_score += it->score;
        mean_pair_dy += it->center_dy_px;
    }
    mean_pair_score /= static_cast<float>(length);
    mean_pair_dy /= static_cast<float>(length);

    float gap_consistency    = 1.0f;
    float vertical_alignment = 1.0f;

    if (length >= 2) {
        std::vector<float> gaps;
        gaps.reserve(length - 1);
        for (size_t i = start + 1; i < start + length; ++i) {
            gaps.push_back(sorted_pairs[i].midpoint_px.x - sorted_pairs[i - 1].midpoint_px.x);
        }

        const float mean_gap =
            std::accumulate(gaps.begin(), gaps.end(), 0.0f) / static_cast<float>(gaps.size());
        if (mean_gap <= 1e-3f) {
            return 0.0f;
        }

        float sq_sum = 0.0f;
        for (const float gap : gaps) {
            const float delta = gap - mean_gap;
            sq_sum += delta * delta;
        }
        const float std_gap = std::sqrt(sq_sum / static_cast<float>(gaps.size()));
        const float gap_cv  = std_gap / mean_gap;
        if (gap_cv > static_cast<float>(config.max_gap_cv)) {
            return 0.0f;
        }
        gap_consistency = std::max(0.0f, 1.0f - gap_cv / static_cast<float>(config.max_gap_cv));
    }

    if (mean_pair_dy > 1e-3f) {
        float max_delta = 0.0f;
        for (auto it = group_begin; it != group_end; ++it) {
            max_delta = std::max(max_delta, std::abs(it->center_dy_px - mean_pair_dy));
        }
        vertical_alignment =
            std::max(0.0f, 1.0f - max_delta / std::max(1.0f, mean_pair_dy * 0.35f));
    }

    return std::clamp(
        0.55f * mean_pair_score + 0.30f * gap_consistency + 0.15f * vertical_alignment, 0.0f, 1.0f);
}

[[nodiscard]] inline std::vector<LdmMeshCandidate> build_preliminary_mesh_candidates(
    std::vector<LightPair> sorted_pairs, const LdmDetectorConfig& config) {
    std::sort(sorted_pairs.begin(), sorted_pairs.end(), [](const LightPair& a, const LightPair& b) {
        return a.midpoint_px.x < b.midpoint_px.x;
    });

    std::vector<LdmMeshCandidate> candidates;
    if (sorted_pairs.empty()) {
        return candidates;
    }

    const size_t max_pairs = static_cast<size_t>(std::max(1, config.max_pairs_per_candidate));
    const size_t limit     = std::min(sorted_pairs.size(), max_pairs);
    for (size_t length = limit; length >= 1; --length) {
        for (size_t start = 0; start + length <= sorted_pairs.size(); ++start) {
            LdmMeshCandidate candidate;
            candidate.preliminary_score = pair_group_score(sorted_pairs, start, length, config);
            if (candidate.preliminary_score <= 0.0f) {
                continue;
            }

            candidate.pair_indices.reserve(length);
            for (size_t i = 0; i < length; ++i) {
                const auto& pair = sorted_pairs[start + i];
                candidate.pair_indices.push_back(static_cast<int>(start + i));
                candidate.estimated_center_image_px += pair.midpoint_px;
            }
            candidate.estimated_center_image_px *= (1.0f / static_cast<float>(length));
            candidates.push_back(std::move(candidate));
        }

        if (length == 1) {
            break;
        }
    }

    std::sort(
        candidates.begin(), candidates.end(),
        [](const LdmMeshCandidate& a, const LdmMeshCandidate& b) {
            if (a.pair_indices.size() != b.pair_indices.size()) {
                return a.pair_indices.size() > b.pair_indices.size();
            }
            return a.preliminary_score > b.preliminary_score;
        });
    return candidates;
}

} // namespace fcs::L2::ldm
