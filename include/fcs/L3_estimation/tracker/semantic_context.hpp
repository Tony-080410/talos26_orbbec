#pragma once

#include <array>
#include <cmath>

namespace fcs::L3 {

/// Armor class types for semantic matching
enum class ArmorClass : int {
    Red     = 0,
    Blue    = 1,
    Grey    = 2, // Balance
    Hero    = 3,
    Sentry  = 4, // Outpost
    Unknown = 5
};

/// Semantic context for armor matching
///
/// Provides semantic information to break symmetry in ambiguous geometric cases.
/// Uses class transition probabilities and detector confidence to refine matching.
class SemanticContext {
public:
    SemanticContext() = default;

    /// Get the expected armor class for a given armor ID
    ///
    /// @param armor_id The armor ID (0-3 for robots)
    /// @return Expected armor class
    [[nodiscard]] ArmorClass get_expected_class(int armor_id) const {
        // Default: assume first 2 armors are red (front), last 2 are grey (back)
        // This can be configured based on target type
        if (armor_id < 2) {
            return ArmorClass::Red;
        }
        return ArmorClass::Grey;
    }

    /// Get log probability of detected class given true class
    ///
    /// @param detected_class The class detected by the vision system
    /// @param true_class The expected true class
    /// @return Log probability P(detected | true)
    [[nodiscard]] double
        log_class_prob(ArmorClass detected_class, ArmorClass true_class) const noexcept {
        // P(detected | true) transition matrix (from config/statistics)
        // TODO: Make this configurable
        static constexpr double CLASS_TRANSITION[6][6] = {
            //     Red   Blue  Grey  Hero Sentry Unknown
            {0.95, 0.02, 0.02, 0.01, 0.00, 0.00}, // Red
            {0.02, 0.95, 0.02, 0.01, 0.00, 0.00}, // Blue
            {0.05, 0.05, 0.85, 0.05, 0.00, 0.00}, // Grey (balance)
            {0.00, 0.00, 0.10, 0.90, 0.00, 0.00}, // Hero
            {0.00, 0.00, 0.00, 0.00, 1.00, 0.00}, // Sentry (outpost)
            {0.20, 0.20, 0.20, 0.20, 0.20, 0.00}, // Unknown
        };

        const int d = static_cast<int>(detected_class);
        const int t = static_cast<int>(true_class);

        return std::log(CLASS_TRANSITION[t][d] + 1e-10);
    }

    /// Check if semantic context is enabled
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    /// Enable/disable semantic context
    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }

private:
    bool enabled_{false};
};

} // namespace fcs::L3
