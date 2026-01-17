#pragma once

#include "tiny_api.hpp"

#include <utility>

namespace fcs::L4 {

/// RAII wrapper for TinySolver resource management
///
/// Provides automatic cleanup of TinySolver resources following RAII principles.
/// Non-copyable but movable to transfer ownership.
class SolverHandle {
public:
    /// Default constructor creates null handle
    SolverHandle() noexcept = default;

    /// Destructor automatically cleanup solver
    ~SolverHandle() noexcept { cleanup(); }

    // Non-copyable
    SolverHandle(const SolverHandle&)            = delete;
    SolverHandle& operator=(const SolverHandle&) = delete;

    // Movable
    SolverHandle(SolverHandle&& other) noexcept
        : solver_(other.solver_) {
        other.solver_ = nullptr;
    }

    SolverHandle& operator=(SolverHandle&& other) noexcept {
        if (this != &other) {
            cleanup();
            solver_       = other.solver_;
            other.solver_ = nullptr;
        }
        return *this;
    }

    /// Get raw pointer (non-owning)
    [[nodiscard]] TinySolver* get() const noexcept { return solver_; }

    /// Get raw pointer (non-owning)
    [[nodiscard]] TinySolver* operator->() const noexcept { return solver_; }

    /// Explicit conversion to bool
    [[nodiscard]] explicit operator bool() const noexcept { return solver_ != nullptr; }

    /// Release ownership without cleanup
    [[nodiscard]] TinySolver* release() noexcept {
        auto* tmp = solver_;
        solver_   = nullptr;
        return tmp;
    }

    /// Reset with new solver (takes ownership)
    void reset(TinySolver* solver = nullptr) noexcept {
        cleanup();
        solver_ = solver;
    }

    /// Swap with another handle
    void swap(SolverHandle& other) noexcept {
        auto* tmp     = solver_;
        solver_       = other.solver_;
        other.solver_ = tmp;
    }

private:
    void cleanup() noexcept {
        if (solver_) {
            tiny_cleanup(solver_);
            solver_ = nullptr;
        }
    }

    TinySolver* solver_{nullptr};
};

} // namespace fcs::L4
