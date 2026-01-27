#pragma once

#include <Eigen/Dense>

namespace uni_kf::redact::factorization {

/// In-place rank-1 update/downdate of a LOWER-triangular Cholesky factor L.
///
/// Updates L such that:
///   L_new * L_new^T = L * L^T + sign * x * x^T
/// where sign is +1 (update) or -1 (downdate).
///
/// **Algorithm**:
/// Uses hyperbolic rotations (Givens rotations) to update the Cholesky factor
/// in O(n²) time without recomputing the full decomposition.
///
/// **Reference**:
/// Gill, P. E., Golub, G. H., Murray, W., & Saunders, M. A. (1974).
/// "Methods for modifying matrix factorizations."
/// *Mathematics of Computation*, 28(126), 505-535.
/// DOI: [10.2307/2005923](https://doi.org/10.2307/2005923)
/// - Section 3: Rank-1 modifications of Cholesky factors
/// - Equations (3.4)-(3.6): Update/downdate algorithms
///
/// Also see:
/// - Brent, R. P. (1975). "A note on downdating the Cholesky factorization."
///   [PDF](https://maths-people.anu.edu.au/~brent/pd/rpb095i.pdf)
///
/// @param L Lower-triangular Cholesky factor (modified in-place)
/// @param x Vector to add/remove
/// @param sign +1 for update (LL^T + xx^T), -1 for downdate (LL^T - xx^T)
/// @return true if successful, false if downdate would make matrix non-SPD
inline bool cholupdate_lower_inplace(Eigen::MatrixXd& L, Eigen::VectorXd x, int sign) {
    const int n = static_cast<int>(L.rows());
    if (L.cols() != n || x.size() != n) {
        return false;
    }
    if (sign != 1 && sign != -1) {
        return false;
    }

    for (int k = 0; k < n; ++k) {
        const double lkk = L(k, k);
        const double xk  = x(k);

        const double tmp = lkk * lkk + static_cast<double>(sign) * xk * xk;
        if (tmp <= 0.0) {
            return false;
        }

        const double r = std::sqrt(tmp);
        const double c = r / lkk;
        const double s = xk / lkk;

        L(k, k) = r;

        for (int i = k + 1; i < n; ++i) {
            const double lik = L(i, k);
            const double xi  = x(i);

            // Update L(i,k) first (depends on sign), then update x(i).
            const double lik_new = (sign == 1) ? ((lik + s * xi) / c) : ((lik - s * xi) / c);
            L(i, k)              = lik_new;
            x(i)                 = c * xi - s * lik_new;
        }
    }

    // Keep upper triangle clean (optional, but helps debugging/printing)
    L.triangularView<Eigen::StrictlyUpper>().setZero();
    return true;
}

} // namespace uni_kf::redact::factorization
