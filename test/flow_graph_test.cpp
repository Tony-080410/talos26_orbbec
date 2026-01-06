// Force enable flow graph for testing
#define ENABLE_FLOW_GRAPH 1

#include <gtest/gtest.h>

#include <string>

#include "scheduler/primitives/channel.hpp"
#include "scheduler/primitives/flow_graph.hpp"

using namespace primitive;

namespace {

struct TestData {
    int value{0};
};

} // namespace

class FlowGraphTest : public testing::Test {
protected:
    void SetUp() override { global_flow_graph().clear(); }
};

TEST_F(FlowGraphTest, ChannelCreateRecordsNothing) {
    auto ch = make_spmc_channel<TestData>();

    // create() alone doesn't add edges, only nodes implicitly when edges are added
    EXPECT_EQ(global_flow_graph().edge_count(), 0);
}

TEST_F(FlowGraphTest, CloneReaderRecordsEdge) {
    const auto ch = make_spmc_channel<TestData>();
    auto reader   = ch.clone_reader();

    EXPECT_EQ(global_flow_graph().edge_count(), 1);
    EXPECT_EQ(global_flow_graph().node_count(), 2);
}

TEST_F(FlowGraphTest, MultipleReadersCreateMultipleEdges) {
    const auto ch = make_spmc_channel<TestData>();

    auto r1 = ch.clone_reader();
    auto r2 = ch.clone_reader();
    auto r3 = ch.clone_reader();

    EXPECT_EQ(global_flow_graph().edge_count(), 3);
    // Producer + 3 consumers = 4 nodes
    EXPECT_EQ(global_flow_graph().node_count(), 4);
}

TEST_F(FlowGraphTest, TopologicalSortWorks) {
    const auto ch1 = make_spmc_channel<TestData>(); // source
    auto r1        = ch1.clone_reader();            // mid

    const auto ch2 = make_spmc_channel<int>();      // mid
    auto r2        = ch2.clone_reader();            // sink

    const auto [order, has_cycle] = global_flow_graph().topological_sort();

    EXPECT_FALSE(has_cycle);
    EXPECT_EQ(order.size(), global_flow_graph().node_count());
}

TEST_F(FlowGraphTest, SourcesAndSinks) {
    const auto ch = make_spmc_channel<TestData>();

    auto r1 = ch.clone_reader();
    auto r2 = ch.clone_reader();

    const auto sources = global_flow_graph().sources();
    const auto sinks   = global_flow_graph().sinks();

    EXPECT_EQ(sources.size(), 1);                   // producer
    EXPECT_EQ(sinks.size(), 2);                     // two consumers
}

TEST_F(FlowGraphTest, ExportToDot) {
    const auto ch = make_spmc_channel<TestData>();
    auto reader   = ch.clone_reader();

    const auto dot = global_flow_graph().to_dot("TestGraph");

    EXPECT_TRUE(dot.find("digraph TestGraph") != std::string::npos);
    EXPECT_TRUE(dot.find("->") != std::string::npos);
}

TEST_F(FlowGraphTest, ExportToJson) {
    const auto ch = make_spmc_channel<TestData>();
    auto reader   = ch.clone_reader();

    const auto json = global_flow_graph().to_json();

    EXPECT_TRUE(json.find("\"nodes\"") != std::string::npos);
    EXPECT_TRUE(json.find("\"edges\"") != std::string::npos);
}

TEST_F(FlowGraphTest, ChannelReadWriteWorks) {
    auto ch     = make_spmc_channel<TestData>();
    auto reader = ch.clone_reader();

    ch.write(TestData{42});

    const auto data = reader.read();
    ASSERT_TRUE(data.has_value());
    EXPECT_EQ(data->value, 42);
}

TEST_F(FlowGraphTest, SplitWorks) {
    auto ch               = make_spmc_channel<TestData>();
    auto [writer, reader] = ch.split();

    writer.write(TestData{123});

    const auto data = reader.read();
    ASSERT_TRUE(data.has_value());
    EXPECT_EQ(data->value, 123);

    EXPECT_EQ(global_flow_graph().edge_count(), 1);
}

// Simulate a real pipeline
TEST_F(FlowGraphTest, PipelineTopology) {
    // camera -> detector -> tracker -> predictor
    //                    \-> visualizer

    auto camera_out  = make_spmc_channel<int>();
    auto detector_in = camera_out.clone_reader();

    auto detector_out = make_spmc_channel<int>();
    auto tracker_in   = detector_out.clone_reader();
    auto viz_in       = detector_out.clone_reader();

    auto tracker_out  = make_spmc_channel<int>();
    auto predictor_in = tracker_out.clone_reader();

    const auto& g = global_flow_graph();

    // 4 channels, 4 edges
    EXPECT_EQ(g.edge_count(), 4);

    // Check no cycles
    EXPECT_FALSE(g.has_cycle());

    const auto sources = g.sources();
    const auto sinks   = g.sinks();

    // 3 source channels
    EXPECT_EQ(sources.size(), 3);
    // 4 terminal consumers
    EXPECT_EQ(sinks.size(), 4);
}

// ============================================================================
// SPSC Channel tests
// ============================================================================

TEST_F(FlowGraphTest, SpscChannelWorks) {
    auto ch = make_spsc_channel<TestData>();

    // SPSC: must use split() or take_reader(), no clone_reader()
    auto [writer, reader] = ch.split();

    writer.write(TestData{99});

    const auto data = reader.read();
    ASSERT_TRUE(data.has_value());
    EXPECT_EQ(data->value, 99);

    EXPECT_EQ(global_flow_graph().edge_count(), 1);
}

TEST_F(FlowGraphTest, SpscBorrowMutPublish) {
    auto ch               = make_spsc_channel<TestData>();
    auto [writer, reader] = ch.split();

    // Use borrow_mut + publish pattern
    writer.borrow_mut().value = 42;
    writer.publish();

    const auto data = reader.read();
    ASSERT_TRUE(data.has_value());
    EXPECT_EQ(data->value, 42);
}

TEST_F(FlowGraphTest, SpscReaderIsMoveOnly) {
    // Compile-time check: SPSC reader should not be copyable
    static_assert(!std::is_copy_constructible_v<SpscChannel<int>::Reader>);
    static_assert(!std::is_copy_assignable_v<SpscChannel<int>::Reader>);
    static_assert(std::is_move_constructible_v<SpscChannel<int>::Reader>);
}

TEST_F(FlowGraphTest, SpmcReaderIsCopyable) {
    // Compile-time check: SPMC reader should be copyable
    static_assert(std::is_copy_constructible_v<SpmcChannel<int>::Reader>);
    static_assert(std::is_copy_assignable_v<SpmcChannel<int>::Reader>);
}

TEST_F(FlowGraphTest, MixedPipeline) {
    // SPSC for high-throughput, SPMC for fan-out
    auto spsc_ch                           = make_spsc_channel<int>(); // sensor -> processor
    auto [sensor_writer, processor_reader] = spsc_ch.split();

    const auto spmc_ch = make_spmc_channel<int>();  // processor -> multiple consumers
    auto consumer1     = spmc_ch.clone_reader();
    auto consumer2     = spmc_ch.clone_reader();

    EXPECT_EQ(global_flow_graph().edge_count(), 3); // 1 SPSC + 2 SPMC
}
