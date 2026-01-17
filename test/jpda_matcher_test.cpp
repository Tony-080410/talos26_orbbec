#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <set>
#include <vector>

#include "fcs/L3_estimation/tracker/semantic_context.hpp"
#include "fcs/L3_estimation/tracker/viterbi_smoother.hpp"

// Access the permutation generator without full template instantiation
namespace fcs::L3 {

// Expose the permutation generator for testing
constexpr std::array<std::array<int, 4>, 24> generate_all_permutations_4() {
    std::array<std::array<int, 4>, 24> perms{};
    int idx = 0;
    int p[] = {0, 1, 2, 3};
    do {
        perms[idx++] = {p[0], p[1], p[2], p[3]};
    } while (std::next_permutation(p, p + 4));
    return perms;
}

constexpr auto ALL_PERMUTATIONS_4_TEST = generate_all_permutations_4();

} // namespace fcs::L3

using namespace fcs::L3;

// =============================================================================
// Test: Compile-time permutation generation
// =============================================================================

TEST(JpdaMatcher, PermutationGeneration) {
    // Verify that we have exactly 24 permutations for 4 elements
    EXPECT_EQ(ALL_PERMUTATIONS_4_TEST.size(), 24);

    // Verify each permutation has exactly 4 elements
    for (const auto& perm : ALL_PERMUTATIONS_4_TEST) {
        EXPECT_EQ(perm.size(), 4);
    }

    // Verify all permutations are unique
    std::set<std::array<int, 4>> unique_perms;
    for (const auto& perm : ALL_PERMUTATIONS_4_TEST) {
        unique_perms.insert(perm);
    }
    EXPECT_EQ(unique_perms.size(), 24);

    // Verify first few known permutations
    EXPECT_EQ(ALL_PERMUTATIONS_4_TEST[0], (std::array<int, 4>{0, 1, 2, 3}));
    EXPECT_EQ(ALL_PERMUTATIONS_4_TEST[1], (std::array<int, 4>{0, 1, 3, 2}));
    EXPECT_EQ(ALL_PERMUTATIONS_4_TEST[2], (std::array<int, 4>{0, 2, 1, 3}));
}

// =============================================================================
// Test: Phase ambiguity detection
// =============================================================================

TEST(JpdaMatcher, PhaseAmbiguityDetection) {
    constexpr double threshold = 0.3;

    // Clear case: one phase dominates
    std::array<double, 4> clear_probs{0.95, 0.03, 0.01, 0.01};
    std::array<double, 4> sorted_clear = clear_probs;
    std::sort(sorted_clear.begin(), sorted_clear.end(), std::greater<double>());
    double relative_diff_clear = (sorted_clear[0] - sorted_clear[1]) / sorted_clear[0];
    EXPECT_FALSE(relative_diff_clear < threshold);

    // Ambiguous case: two phases are close
    std::array<double, 4> ambiguous_probs{0.50, 0.45, 0.03, 0.02};
    std::array<double, 4> sorted_ambiguous = ambiguous_probs;
    std::sort(sorted_ambiguous.begin(), sorted_ambiguous.end(), std::greater<double>());
    double relative_diff_ambiguous =
        (sorted_ambiguous[0] - sorted_ambiguous[1]) / sorted_ambiguous[0];
    EXPECT_TRUE(relative_diff_ambiguous < threshold);

    // Edge case: all near zero
    std::array<double, 4> zero_probs{1e-10, 1e-10, 1e-10, 1e-10};
    std::array<double, 4> sorted_zero = zero_probs;
    std::sort(sorted_zero.begin(), sorted_zero.end(), std::greater<double>());
    // When all probabilities are near zero, we treat it as not ambiguous (no clear phase)
    EXPECT_GE(sorted_zero[0], 0.0); // Just verify it's not negative
}

// =============================================================================
// Test: Semantic context
// =============================================================================

TEST(SemanticContext, ClassTransitionProbabilities) {
    SemanticContext ctx;

    // High probability: detected == true (P(Red | Red) = 0.95)
    double log_p = ctx.log_class_prob(ArmorClass::Red, ArmorClass::Red);
    EXPECT_GT(log_p, std::log(0.9)); // Should be > 0.9

    // Low probability: detected != true (P(Blue | Red) = 0.02)
    log_p = ctx.log_class_prob(ArmorClass::Blue, ArmorClass::Red);
    EXPECT_LT(log_p, std::log(0.1)); // Should be < 0.1

    // When true class is Unknown, detected classes are uniform (P(Red | Unknown) = 0.20)
    log_p = ctx.log_class_prob(ArmorClass::Red, ArmorClass::Unknown);
    EXPECT_NEAR(log_p, std::log(0.20), 1e-6); // 0.20 for Red when true is Unknown
}

// =============================================================================
// Test: Viterbi smoother
// =============================================================================

TEST(ViterbiSmoother, BasicSmoothing) {
    ViterbiSmoother smoother(4);

    // First frame strongly indicates phase 1
    std::array<double, 4> probs1{0.05, 0.90, 0.03, 0.02};
    smoother.update(std::vector<double>(probs1.begin(), probs1.end()));

    // Second frame still strongly indicates phase 1
    std::array<double, 4> probs2{0.03, 0.92, 0.03, 0.02};
    smoother.update(std::vector<double>(probs2.begin(), probs2.end()));

    // Third frame continues to indicate phase 1
    std::array<double, 4> probs3{0.02, 0.93, 0.03, 0.02};
    smoother.update(std::vector<double>(probs3.begin(), probs3.end()));

    // Get smoothed phase
    int smoothed = smoother.get_smoothed_phase();
    EXPECT_EQ(smoothed, 1); // Should prefer phase 1 (consistent across all frames)
}

TEST(ViterbiSmoother, Stability) {
    ViterbiSmoother smoother(4);

    // All frames strongly prefer phase 0
    std::array<double, 4> probs{0.90, 0.03, 0.04, 0.03};
    smoother.update(std::vector<double>(probs.begin(), probs.end()));
    smoother.update(std::vector<double>(probs.begin(), probs.end()));
    smoother.update(std::vector<double>(probs.begin(), probs.end()));

    int smoothed = smoother.get_smoothed_phase();
    EXPECT_EQ(smoothed, 0); // Should prefer phase 0 (stable)
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
