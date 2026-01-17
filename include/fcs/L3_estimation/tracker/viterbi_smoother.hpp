#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fcs::L3 {
class ViterbiSmoother {
public:
    static constexpr int WINDOW_SIZE = 3;
    static constexpr double LOG_EPS  = 1e-10;

    explicit ViterbiSmoother(int n_phases = 4) noexcept
        : n_phases_(n_phases) {}

    int size() const noexcept { return n_phases_; }

    void set_switch_lambda(double lambda) noexcept { switch_lambda_ = lambda; }

    void reset() noexcept {
        emissions_.clear();
        last_smoothed_phase_ = 0;
    }

    void update(const std::vector<double>& phase_probs) noexcept {
        if (phase_probs.empty())
            return;

        if (static_cast<int>(phase_probs.size()) != n_phases_) {
            n_phases_ = static_cast<int>(phase_probs.size());
            emissions_.clear();
        }

        // push emission (log)
        std::vector<double> logp(n_phases_, 0.0);
        for (int p = 0; p < n_phases_; ++p) {
            logp[p] = std::log(phase_probs[p] + LOG_EPS);
        }
        emissions_.push_back(std::move(logp));
        if (static_cast<int>(emissions_.size()) > WINDOW_SIZE) {
            emissions_.erase(emissions_.begin());
        }

        // DP over the window
        const int T = static_cast<int>(emissions_.size());
        std::vector<std::vector<double>> dp(T, std::vector<double>(n_phases_, -INFINITY));
        std::vector<std::vector<int>> back(T, std::vector<int>(n_phases_, 0));

        // init: uniform prior
        for (int p = 0; p < n_phases_; ++p)
            dp[0][p] = emissions_[0][p];

        for (int t = 1; t < T; ++t) {
            for (int p = 0; p < n_phases_; ++p) {
                double best   = -INFINITY;
                int best_prev = 0;
                for (int q = 0; q < n_phases_; ++q) {
                    const int d       = cyc_dist(p, q, n_phases_);
                    const double cand = dp[t - 1][q] + emissions_[t][p] - switch_lambda_ * d;
                    if (cand > best) {
                        best      = cand;
                        best_prev = q;
                    }
                }
                dp[t][p]   = best;
                back[t][p] = best_prev;
            }
        }

        // backtrack from last time
        int best_last = 0;
        for (int p = 1; p < n_phases_; ++p) {
            if (dp[T - 1][p] > dp[T - 1][best_last])
                best_last = p;
        }

        // fixed-lag: output middle frame if possible (lag=1)
        if (T >= 2) {
            last_smoothed_phase_ = back[T - 1][best_last]; // phase at time (T-2)
        } else {
            last_smoothed_phase_ = best_last;
        }
    }

    int get_smoothed_phase() const noexcept { return last_smoothed_phase_; }

private:
    static int cyc_dist(int a, int b, int n) noexcept {
        int d = std::abs(a - b);
        return std::min(d, n - d);
    }

    int n_phases_{4};
    double switch_lambda_{0.1};
    std::vector<std::vector<double>> emissions_;
    int last_smoothed_phase_{0};
};

} // namespace fcs::L3
