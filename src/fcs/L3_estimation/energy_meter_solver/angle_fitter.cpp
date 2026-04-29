#include "L3_estimation/energy_meter_solver/angle_fitter.hpp"

#include <algorithm>
#include <cmath>

namespace energy_meter {

AngleFitter::AngleFitter(Config config)
    : config_(config) {
    params_ = config_.initial_params;
}

AngleFitter::~AngleFitter() = default;

void AngleFitter::addObservation(double time, double abs_angle) {
    std::unique_lock lock(data_mutex_);
    fit_data_.emplace_back(time, abs_angle);

    // Trim oldest half when exceeding max size (same as rm_power_rune)
    if (static_cast<int>(fit_data_.size()) > config_.max_fit_data_size) {
        auto mid = fit_data_.begin() + static_cast<ptrdiff_t>(fit_data_.size() / 2);
        fit_data_.erase(fit_data_.begin(), mid);
    }
}

bool AngleFitter::fitBigRune() {
    std::shared_lock lock(data_mutex_);

    if (static_cast<int>(fit_data_.size()) < config_.min_fit_data_size) {
        return false;
    }

    // Determine convexity when data is still small
    if (static_cast<int>(fit_data_.size()) < 2 * config_.min_fit_data_size) {
        convexity_ = detectConvexity(fit_data_);
    }

    auto data_copy = fit_data_;
    lock.unlock();

    auto new_params = ransacFitting(convexity_);

    // Compute RMSE on the original data
    double sum_squared_error = 0.0;
    for (const auto& pt : data_copy) {
        const double predicted = getAngleBig(pt.first, new_params);
        const double residual  = pt.second - predicted;
        sum_squared_error += residual * residual;
    }
    const double rmse = std::sqrt(sum_squared_error / static_cast<double>(data_copy.size()));

    std::unique_lock mlock(model_mutex_);
    params_       = new_params;
    params_valid_ = true;
    fit_error_    = rmse;
    return true;
}

bool AngleFitter::fitSmallRune() {
    std::unique_lock mlock(model_mutex_);
    params_valid_ = true;
    return true;
}

auto AngleFitter::getModel() const -> AngleModel {
    std::shared_lock mlock(model_mutex_);
    AngleModel m;
    m.a                = params_[0];
    m.omega            = params_[1];
    m.t0               = params_[2];
    m.b                = params_[3];
    m.c                = params_[4];
    m.rotation_speed   = config_.small_rune_rotation_speed;
    m.is_valid         = params_valid_;
    m.data_points_used = [&] {
        std::shared_lock dlock(data_mutex_);
        return static_cast<int>(fit_data_.size());
    }();
    mlock.unlock();
    // fit_error_ is read-only after fitting, safe to access without lock
    m.fit_error = fit_error_;
    return m;
}

auto AngleFitter::getDirection() const -> Direction { return direction_; }

auto AngleFitter::getDataPointCount() const -> int {
    std::shared_lock lock(data_mutex_);
    return static_cast<int>(fit_data_.size());
}

bool AngleFitter::isModelValid() const {
    std::shared_lock lock(model_mutex_);
    return params_valid_;
}

void AngleFitter::reset() {
    {
        std::unique_lock lock(data_mutex_);
        fit_data_.clear();
    }
    {
        std::unique_lock lock(model_mutex_);
        params_       = config_.initial_params;
        params_valid_ = false;
        convexity_    = Convexity::UNKNOWN;
        fit_error_    = 0.0;
    }
    direction_ = Direction::UNKNOWN;
    direction_data_.clear();
}

// double AngleFitter::predictDeltaAngle(double current_time, double future_time) const {
//     std::shared_lock lock(model_mutex_);
//     return getAngleBig(future_time, params_) - getAngleBig(current_time, params_);
// }

void AngleFitter::updateDirection(double continuous_angle) {
    if (direction_ != Direction::UNKNOWN && direction_ != Direction::STABLE) {
        return;
    }

    direction_data_.push_back(continuous_angle);
    if (static_cast<int>(direction_data_.size()) < config_.direction_threshold) {
        return;
    }

    int stable = 0, anti = 0, clockwise = 0;
    size_t half = direction_data_.size() / 2;
    for (size_t i = 0; i < half; ++i) {
        double diff = direction_data_[i + half] - direction_data_[i];
        if (diff > config_.direction_angle_threshold) {
            clockwise++;
        } else if (diff < -config_.direction_angle_threshold) {
            anti++;
        } else {
            stable++;
        }
    }

    int best = std::max({stable, clockwise, anti});
    if (best == clockwise) {
        direction_ = Direction::CLOCKWISE;
    } else if (best == anti) {
        direction_ = Direction::ANTI_CLOCKWISE;
    } else {
        direction_ = Direction::STABLE;
    }
}

// ============================================================================
// RANSAC fitting (rm_power_rune semantics)
// ============================================================================

std::array<double, 5> AngleFitter::ransacFitting(Convexity convexity) {
    std::shared_lock lock(data_mutex_);

    // Work on a local copy so the caller's vector is untouched.
    std::vector<std::pair<double, double>> inliers = fit_data_;
    std::vector<std::pair<double, double>> outliers;
    lock.unlock();

    const int iter_times = static_cast<int>(inliers.size()) < config_.few_data_threshold
                             ? config_.ransac_iterations_few
                             : config_.ransac_iterations_many;

    std::array<double, 5> params = config_.initial_params;

    for (int iter = 0; iter < iter_times; ++iter) {
        // Sub-sample for large datasets
        std::vector<std::pair<double, double>> sample;
        if (static_cast<int>(inliers.size()) > config_.few_data_threshold) {
            auto tail = inliers.end() - config_.outlier_keep_tail;
            std::shuffle(inliers.begin(), tail, random_engine_);
            auto start = inliers.end() - config_.inlier_sample_size;
            if (start < inliers.begin()) {
                start = inliers.begin();
            }
            sample.assign(start, inliers.end());
        } else {
            sample = inliers;
        }

        params = leastSquareEstimate(sample, params, convexity);

        // Compute errors on inliers
        std::vector<double> errors;
        errors.reserve(inliers.size());
        for (const auto& pt : inliers) {
            errors.push_back(std::abs(pt.second - getAngleBig(pt.first, params)));
        }

        // Inlier/outlier exchange for large data (rm_power_rune semantics)
        if (static_cast<int>(inliers.size()) > config_.large_data_threshold) {
            std::sort(errors.begin(), errors.end());
            size_t idx = std::min(
                static_cast<size_t>(errors.size() * config_.inlier_percentile), errors.size() - 1);
            double threshold = errors[idx];

            // Move bad inliers → outliers (iterate backwards to keep indices valid)
            for (int i = static_cast<int>(inliers.size()) - config_.outlier_keep_tail - 1; i >= 0;
                 --i) {
                if (std::abs(inliers[i].second - getAngleBig(inliers[i].first, params))
                    > threshold) {
                    outliers.push_back(std::move(inliers[i]));
                    inliers.erase(inliers.begin() + i);
                }
            }

            // Move good outliers → inliers (iterate backwards)
            for (int i = static_cast<int>(outliers.size()) - 1; i >= 0; --i) {
                if (std::abs(outliers[i].second - getAngleBig(outliers[i].first, params))
                    <= threshold) {
                    inliers.push_back(std::move(outliers[i]));
                    outliers.erase(outliers.begin() + i);
                }
            }
        }
    }

    return params;
}

// ============================================================================
// Ceres least-squares estimation (rm_power_rune semantics)
// ============================================================================

std::array<double, 5> AngleFitter::leastSquareEstimate(
    const std::vector<std::pair<double, double>>& points,
    const std::array<double, 5>& initial_params, Convexity convexity) {
    // 使用硬约束: a + b = 2.090，即 b = 2.090 - a
    // 参数从5个减少到4个: [a, w, t0, c]
    std::array<double, 4> constrained_params;
    constrained_params[0] = initial_params[0]; // a
    constrained_params[1] = initial_params[1]; // w
    constrained_params[2] = initial_params[2]; // t0
    constrained_params[3] = initial_params[4]; // c

    ceres::Problem problem;

    // Data residuals with SoftLOne robust loss (使用硬约束版本)
    for (const auto& pt : points) {
        auto* cf = new CostFunctorConstrained(pt.first, pt.second, config_.amplitude_offset_sum);
        auto* lf = new ceres::SoftLOneLoss(0.1);
        problem.AddResidualBlock(cf, lf, constrained_params.data());
    }

    // Regularisation weights depend on data size
    const std::array<double, 3>& omega = static_cast<int>(points.size()) < 100
                                           ? config_.regularization_few
                                           : config_.regularization_many;

    // Physical parameter bounds for a (b 由约束自动确定)
    problem.SetParameterLowerBound(constrained_params.data(), 0, config_.amplitude_lower);
    problem.SetParameterUpperBound(constrained_params.data(), 0, config_.amplitude_upper);
    problem.SetParameterLowerBound(constrained_params.data(), 1, config_.omega_lower);
    problem.SetParameterUpperBound(constrained_params.data(), 1, config_.omega_upper);

    // Convexity-based bounds for phase (t0) on small data
    if (static_cast<int>(points.size()) < 100) {
        if (convexity == Convexity::CONCAVE) {
            problem.SetParameterUpperBound(
                constrained_params.data(), 2, config_.convex_concave_t0_upper);
            problem.SetParameterLowerBound(
                constrained_params.data(), 2, config_.convex_concave_t0_lower);
        } else if (convexity == Convexity::CONVEX) {
            problem.SetParameterUpperBound(
                constrained_params.data(), 2, config_.convex_convex_t0_upper);
            problem.SetParameterLowerBound(
                constrained_params.data(), 2, config_.convex_convex_t0_lower);
        }
    }

    // Parameter regularisation (只对 a 和 w 做 regularisation)
    problem.AddResidualBlock(
        new CostFunctor1Constrained(constrained_params[0], 0),
        new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[0], ceres::TAKE_OWNERSHIP),
        constrained_params.data());
    problem.AddResidualBlock(
        new CostFunctor1Constrained(constrained_params[1], 1),
        new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[1], ceres::TAKE_OWNERSHIP),
        constrained_params.data());

    ceres::Solver::Options options;
    options.linear_solver_type                = ceres::DENSE_QR;
    options.max_num_iterations                = 50;
    options.minimizer_progress_to_stdout      = false;
    options.check_gradients                   = false;
    options.gradient_check_relative_precision = 1e-4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 转换回5参数格式: [a, w, t0, b, c]
    std::array<double, 5> ret;
    ret[0] = constrained_params[0];                                // a
    ret[1] = constrained_params[1];                                // w
    ret[2] = constrained_params[2];                                // t0
    ret[3] = config_.amplitude_offset_sum - constrained_params[0]; // b = 2.090 - a
    ret[4] = constrained_params[3];                                // c
    return ret;
}

// ============================================================================
// Convexity detection (rm_power_rune semantics)
// ============================================================================

Convexity AngleFitter::detectConvexity(const std::vector<std::pair<double, double>>& data) const {
    if (data.size() < 2) {
        return Convexity::UNKNOWN;
    }

    auto first = data.begin();
    auto last  = data.end() - 1;

    double slope = (last->second - first->second) / (last->first - first->first);
    double offset =
        (first->second * last->first - last->second * first->first) / (last->first - first->first);

    int concave = 0;
    int convex  = 0;
    for (const auto& pt : data) {
        if (slope * pt.first + offset > pt.second) {
            concave++;
        } else {
            convex++;
        }
    }

    const int standard = static_cast<int>(data.size() * config_.convexity_threshold);
    if (concave > standard) {
        return Convexity::CONCAVE;
    }
    if (convex > standard) {
        return Convexity::CONVEX;
    }
    return Convexity::UNKNOWN;
}

} // namespace energy_meter
