// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "integration_test_helper.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace tokenspeed::test {

class LoadBackViaCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 5;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void SetupHostCache() {
        Submit(MakeRequestSpec("r_seed", /*num_pages=*/2, /*start=*/1));
        PlanOnce();
        SendForwardDone("r_seed", {42});
        const ExecutionPlan seed_stream = PlanOnce();
        ASSERT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(seed_stream).empty());
        AckWriteBacks(seed_stream);
        SendFinish("r_seed");
        AckWriteBacks(PlanOnce());
        PlanOnce();

        Submit(MakeRequestSpec("r_fill", /*num_pages=*/3, /*start=*/100));
        PlanOnce();
        SendForwardDone("r_fill", {200});
        AckWriteBacks(PlanOnce());
        SendFinish("r_fill");
        AckWriteBacks(PlanOnce());
        PlanOnce();
    }
};

TEST_F(LoadBackViaCacheTestSuite, LoadBack_TriggeredAfterPrefetchPopulatesHostCache) {
    SetupHostCache();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<LoadBackBatch>(plan);

    bool r1_in_forward = false;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            for (const auto& rid : fwd->request_ids) {
                if (rid == "r1") r1_in_forward = true;
            }
        }
    }
    EXPECT_TRUE(r1_in_forward || !lb.empty())
        << "host cache hit should trigger LoadBack inline or r1 should be in forward";
}

TEST_F(SchedulerTestSuite, LoadBack_NotTriggeredWithoutHostCacheHit) {
    Submit(MakeRequestSpec("r1", 4));
    auto plan = PlanOnce();
    auto lb = ExtractCacheOpsOfKind<LoadBackBatch>(plan);
    EXPECT_TRUE(lb.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_WhenNoRequests) {
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

TEST_F(SchedulerTestSuite, NoCacheOps_PlainRequestNoCacheHit) {
    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    auto cache_ops = ExtractCacheOps(plan);
    EXPECT_TRUE(cache_ops.empty());
}

class DisablePrefixCacheTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(DisablePrefixCacheTestSuite, SamePromptDoesNotReuseDevicePrefix) {
    Submit(MakeRequestSpec("r_seed", 2));
    PlanOnce();
    SendForwardDone("r_seed", {100});
    PlanOnce();
    SendFinish("r_seed");
    PlanOnce();

    Submit(MakeRequestSpec("r1", 2));
    auto plan = PlanOnce();
    const auto& op = plan.Operations()[0];
    auto* fwd = std::get_if<ForwardBatch>(&op);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r1");
    EXPECT_EQ(fwd->extend_prefix_lens[0], 0);
    EXPECT_EQ(fwd->input_lengths[0], 4);
    EXPECT_TRUE(ExtractCacheOpsOfKind<LoadBackBatch>(plan).empty());
}

class StableCandidateOrderingSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        // Force the candidates loop to break after exactly one push so the
        // tiebreaker decides which request wins.
        cfg.max_batch_size = 1;
        return cfg;
    }
};

TEST_F(StableCandidateOrderingSuite, ForwardOperationsFollowSubmissionOrder) {
    // TP-determinism + FIFO: requests_ is a vector in submission order, and
    // the mirrored schedulers receive identical submission sequences -- so
    // when the loop budget admits only a subset, every rank picks the same
    // request, and it is the OLDEST one, whatever its id sorts like.
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan = PlanOnce();
    std::vector<std::string> ids;
    for (const auto& op : plan.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids = fwd->request_ids;
        }
    }
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], "r_ccc") << "first submitted, first scheduled";
}

TEST_F(StableCandidateOrderingSuite, ForwardBatchIsReproducibleAcrossMirroredSchedulers) {
    // Two scheduler instances fed the SAME submission sequence must build
    // the same batch -- the mirrored-ranks invariant. (Different submission
    // orders legitimately differ: FIFO means arrival order is meaningful.)
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan_a = PlanOnce();
    std::vector<std::string> ids_a;
    for (const auto& op : plan_a.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids_a = fwd->request_ids;
        }
    }

    scheduler_ = std::make_unique<Scheduler>(config_);
    Submit(MakeRequestSpec("r_ccc", 2, 300));
    Submit(MakeRequestSpec("r_aaa", 2, 100));
    Submit(MakeRequestSpec("r_bbb", 2, 200));
    auto plan_b = PlanOnce();
    std::vector<std::string> ids_b;
    for (const auto& op : plan_b.Operations()) {
        if (auto* fwd = std::get_if<ForwardBatch>(&op)) {
            ids_b = fwd->request_ids;
        }
    }

    ASSERT_FALSE(ids_a.empty());
    EXPECT_EQ(ids_a, ids_b);
}

class SchedulerKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 3;
        cfg.disable_l2_cache = true;
        cfg.enable_kv_cache_events = true;
        return cfg;
    }
};

class SessionTurnKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_l2_cache = true;
        cfg.enable_kv_cache_events = true;
        return cfg;
    }
};

TEST_F(SessionTurnKvCacheEventTestSuite, SessionTurnIdReuseDoesNotReuseAnotherRequestHashProgress) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first = MakeRequestSpec("turn-1", 2, 1);
    first.session_id = "session-a";
    Submit(first);
    PlanOnce();
    SendForwardDone("turn-1", {50});
    SendFinish("turn-1");

    RequestSpec second = MakeRequestSpec("turn-2", 1, 60);
    second.session_id = "session-a";
    Submit(second);
    PlanOnce();
    SendForwardDone("turn-2", {70});
    SendFinish("turn-2");
    ASSERT_TRUE(scheduler_->CloseSession("session-a"));

    const RequestSpec reused_id = MakeRequestSpec("turn-1", 2, 100);
    Submit(reused_id);
    PlanOnce();
    SendForwardDone("turn-1", {200});
    SendFinish("turn-1");

    const std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    const auto stored = std::find_if(events.begin(), events.end(), [&reused_id](const KvCacheEvent& event) {
        const auto* block = std::get_if<KvBlockStoredEvent>(&event);
        return block != nullptr &&
               block->token_ids == std::vector<std::int32_t>(reused_id.tokens.begin(), reused_id.tokens.begin() + 2);
    });
    ASSERT_NE(stored, events.end());
    const KvBlockStoredEvent& block = std::get<KvBlockStoredEvent>(*stored);
    ASSERT_EQ(block.block_hashes.size(), 1u);
    EXPECT_EQ(block.block_hashes.front(), HashKvBlock(block.token_ids, block.parent_block_hash));
}

TEST_F(SchedulerKvCacheEventTestSuite, PublishesStoredBlockAndDrainsItOnce) {
    const RequestSpec spec = MakeRequestSpec("r1", 1);
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[0]).token_ids, spec.tokens);
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
}

TEST_F(SchedulerKvCacheEventTestSuite, PublishesRemovalWhenAdmissionEvictsBlock) {
    Submit(MakeRequestSpec("seed", 1));
    PlanOnce();
    SendForwardDone("seed", {42});
    PlanOnce();
    SendFinish("seed");

    std::vector<KvCacheEvent> stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 1u);
    const std::uint64_t stored_hash = std::get<KvBlockStoredEvent>(stored[0]).block_hashes.front();

    Submit(MakeRequestSpec("replacement", 1, 100));
    PlanOnce();

    std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(removed[0]));
    EXPECT_EQ(std::get<KvBlockRemovedEvent>(removed[0]).block_hashes, std::vector<std::uint64_t>{stored_hash});
}

class MultiGroupKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 8;
        cfg.disable_l2_cache = true;
        cfg.enable_kv_cache_events = true;
        CacheGroupConfig second = cfg.cache_groups.front();
        second.group_id = "full_attention_1";
        cfg.cache_groups.push_back(std::move(second));
        return cfg;
    }
};

TEST_F(MultiGroupKvCacheEventTestSuite, PublishesOneEventAfterAllGroupsCacheBoundary) {
    Submit(MakeRequestSpec("r1", 1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
}

class SubpageKvCacheEventTestSuite : public SchedulerKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerKvCacheEventTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 4;
        auto& group = cfg.cache_groups.front();
        group.block_granularity = 1;
        group.total_pages = 2 * cfg.device_allocator.total_pages;
        group.cache_blocks_per_lcm_block = 2;
        return cfg;
    }
};

TEST_F(SubpageKvCacheEventTestSuite, PublishesOneEventAfterAllChildBlocksCacheBoundary) {
    const RequestSpec spec = MakeRequestSpec("r1", 1);
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendFinish("r1");

    std::vector<KvCacheEvent> events = scheduler_->DrainKvEvents();
    ASSERT_EQ(events.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(events[0]));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[0]).token_ids, spec.tokens);
}

TEST_F(SubpageKvCacheEventTestSuite, PublishesRemovalWhenAnyChildBlockIsEvicted) {
    Submit(MakeRequestSpec("seed", 2));
    PlanOnce();
    SendForwardDone("seed", {42});
    PlanOnce();
    SendFinish("seed");

    std::vector<KvCacheEvent> stored = scheduler_->DrainKvEvents();
    ASSERT_EQ(stored.size(), 2u);
    const std::uint64_t first_stored_hash = std::get<KvBlockStoredEvent>(stored[0]).block_hashes.front();
    const std::uint64_t second_stored_hash = std::get<KvBlockStoredEvent>(stored[1]).block_hashes.front();

    Submit(MakeRequestSpec("replacement", 2, 100));
    PlanOnce();

    std::vector<KvCacheEvent> removed = scheduler_->DrainKvEvents();
    ASSERT_EQ(removed.size(), 2u);
    std::unordered_set<std::uint64_t> remaining_hashes{first_stored_hash, second_stored_hash};
    for (const KvCacheEvent& event : removed) {
        ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(event));
        const auto& removed_hashes = std::get<KvBlockRemovedEvent>(event).block_hashes;
        ASSERT_EQ(removed_hashes.size(), 1u);
        EXPECT_EQ(remaining_hashes.erase(removed_hashes[0]), 1u);
    }
    EXPECT_TRUE(remaining_hashes.empty());
}

class PrefillRoleKvCacheEventTestSuite : public SchedulerKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerKvCacheEventTestSuite::MakeConfig();
        cfg.role = Role::kP;
        cfg.max_scheduled_tokens = 9;
        cfg.device_allocator.total_pages = 8;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }

    void SendPdSucceeded(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::SucceededEvent{request_id});
        scheduler_->Advance(std::move(event));
    }
};

// A P-role admission publishes the pages the request completed so far, and the
// same Admit may evict to fit the next chunk. When the victim is another
// request's cached copy of a page this request recomputed, the boundary is
// removed and stored again within one Admit: the net state is unchanged, and
// the boundary's descriptor must survive for its later removal.
TEST_F(PrefillRoleKvCacheEventTestSuite, AdmissionThatEvictsAndRepublishesABoundaryKeepsItPublished) {
    RequestSpec seed = MakeRequestSpec("seed", 3);
    seed.tokens.resize(5);
    // dup shares seed's first two pages, then diverges.
    RequestSpec dup = MakeRequestSpec("dup", 5);
    std::copy_n(seed.tokens.begin(), 4, dup.tokens.begin());
    for (std::size_t i = 4; i < dup.tokens.size(); ++i) {
        dup.tokens[i] = 500 + static_cast<std::int32_t>(i);
    }
    Submit({seed, dup});
    SendBootstrapped("seed");
    SendBootstrapped("dup");

    // Neither prompt is cached yet: both compute the shared pages.
    const ExecutionPlan first_round = PlanOnce();
    const ForwardBatch* first = FindForwardBatch(first_round);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->request_ids, (std::vector<std::string>{"seed", "dup"}));
    EXPECT_EQ(first->input_lengths, (std::vector<std::int32_t>{5, 4}));

    // seed's remote decode publishes the shared pages; dup's next chunk does
    // not fit while seed still holds its pages for the PD transfer.
    SendForwardDone("seed", {42});
    const ExecutionPlan second_round = PlanOnce();
    ASSERT_TRUE(second_round.remote_decode.has_value());
    const ForwardBatch* second = FindForwardBatch(second_round);
    ASSERT_NE(second, nullptr);
    EXPECT_TRUE(second->request_ids.empty());
    std::vector<std::uint64_t> shared_hashes;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(event));
        shared_hashes.push_back(std::get<KvBlockStoredEvent>(event).block_hashes.front());
    }
    ASSERT_EQ(shared_hashes.size(), 2u);

    // The PD ACK leaves seed's copies cached and evictable. dup's last chunk
    // (positions 4..9) plus its one-token decode reserve spans pages 2..5,
    // more than are free, so its admission evicts seed's copy of a shared
    // page, then publishes its own.
    SendPdSucceeded("seed");
    const std::int32_t pages_needed = 4;
    ASSERT_LT(scheduler_->CacheGroupAvailablePages("full_attention"), pages_needed);
    const ExecutionPlan third_round = PlanOnce();
    const ForwardBatch* third = FindForwardBatch(third_round);
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(third->request_ids, std::vector<std::string>{"dup"});
    EXPECT_EQ(third->extend_prefix_lens, std::vector<std::int32_t>{4});
    EXPECT_EQ(third->input_lengths, std::vector<std::int32_t>{6});
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());

    // Both shared pages are still published: flushing the cache removes them.
    SendForwardDone("dup", {42});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    SendPdSucceeded("dup");
    scheduler_->DrainKvEvents();  // dup's own later pages
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    std::unordered_set<std::uint64_t> removed;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        ASSERT_TRUE(std::holds_alternative<KvBlockRemovedEvent>(event));
        removed.insert(std::get<KvBlockRemovedEvent>(event).block_hashes.front());
    }
    for (const std::uint64_t hash : shared_hashes) {
        EXPECT_TRUE(removed.contains(hash));
    }
}

// Two cache groups with skewed block granularities, the DeepSeek V4.1 shape:
// comparing position ranks across groups makes eviction strip the fine
// group's copies of a whole boundary range before the coarse group gives up
// a page, leaving those boundaries partially resident.
class SkewedGroupPrefillKvCacheEventTestSuite : public PrefillRoleKvCacheEventTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PrefillRoleKvCacheEventTestSuite::MakeConfig();
        cfg.prefix_granularity = 4;
        cfg.max_scheduled_tokens = 13;
        cfg.device_allocator.total_pages = 9;
        // The fine group comes first so it wins eviction-rank ties against
        // the coarse group's copy of the same boundary.
        CacheGroupConfig fine = cfg.cache_groups.front();
        fine.group_id = "fine_attention";
        fine.block_granularity = 2;
        fine.cache_blocks_per_lcm_block = 2;
        fine.total_pages = 2 * cfg.device_allocator.total_pages;
        fine.transfer_policy = CacheTransferPolicy::FullSuffix;
        CacheGroupConfig& coarse = cfg.cache_groups.front();
        coarse.block_granularity = cfg.prefix_granularity;
        coarse.total_pages = cfg.device_allocator.total_pages;
        cfg.cache_groups.insert(cfg.cache_groups.begin(), std::move(fine));
        return cfg;
    }

    // Runs a fresh prompt to its remote decode and optionally releases its
    // PD pin, without draining any KV events.
    void RunPrefill(const RequestSpec& spec, bool ack) {
        Submit(spec);
        SendBootstrapped(spec.request_id);
        const ExecutionPlan compute_round = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(compute_round);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{spec.request_id});
        SendForwardDone(spec.request_id, {42});
        ASSERT_TRUE(PlanOnce().remote_decode.has_value());
        if (ack) {
            SendPdSucceeded(spec.request_id);
        }
    }
};

// Evictions mark boundaries suffix-first, so a partially resident run whose
// last copies are evicted and then recomputed within one drain window is
// marked child-first. The drain must still emit the recomputed run's Stored
// events parent-first: consumers resolve each event's parent_block_hash
// against what they have already received, so a child arriving ahead of its
// parent is dropped.
TEST_F(SkewedGroupPrefillKvCacheEventTestSuite, RepublishedRunEmitsStoredParentFirst) {
    // seed publishes a three-boundary chain; the evictions below strip it
    // from its suffix.
    RequestSpec seed = MakeRequestSpec("seed", 3);
    RunPrefill(seed, /*ack=*/true);
    std::vector<std::uint64_t> chain_hashes;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        ASSERT_TRUE(std::holds_alternative<KvBlockStoredEvent>(event));
        chain_hashes.push_back(std::get<KvBlockStoredEvent>(event).block_hashes.front());
    }
    ASSERT_EQ(chain_hashes.size(), 3u);

    // filler1's admission evicts across the groups' skewed ranks, taking the
    // chain's fine-group pages and some coarse residues from the back. The
    // first two boundaries come out partially resident: reported removed,
    // descriptors kept.
    RunPrefill(MakeRequestSpec("filler1", 2, 100), /*ack=*/true);
    std::unordered_set<std::uint64_t> removed;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        if (std::holds_alternative<KvBlockRemovedEvent>(event)) {
            removed.insert(std::get<KvBlockRemovedEvent>(event).block_hashes.front());
        }
    }
    EXPECT_TRUE(removed.contains(chain_hashes[0]));
    EXPECT_TRUE(removed.contains(chain_hashes[1]));

    // No drain from here on: the window covers everything below, as one
    // production round covers one admission that evicts and one that
    // publishes. filler2's admission evicts the surviving copies
    // suffix-first, marking the deeper boundary before its parent.
    RunPrefill(MakeRequestSpec("filler2", 2, 200), /*ack=*/true);

    // dup recomputes the chain; its remote decode publishes the new copies.
    RequestSpec dup = MakeRequestSpec("dup", 3);
    dup.tokens = seed.tokens;
    RunPrefill(dup, /*ack=*/false);

    // The drain must order the chain's Stored events parent-first even
    // though the evictions marked it child-first.
    std::vector<std::uint64_t> stored_order;
    for (const KvCacheEvent& event : scheduler_->DrainKvEvents()) {
        if (!std::holds_alternative<KvBlockStoredEvent>(event)) {
            continue;
        }
        const std::uint64_t hash = std::get<KvBlockStoredEvent>(event).block_hashes.front();
        if (std::find(chain_hashes.begin(), chain_hashes.end(), hash) != chain_hashes.end()) {
            stored_order.push_back(hash);
        }
    }
    EXPECT_EQ(stored_order, chain_hashes);
}

TEST_F(SchedulerTestSuite, SubmitRequestsRejectsEmptyTokens) {
    EXPECT_THROW(Submit(RequestSpec{.request_id = "empty"}), std::invalid_argument);
}

TEST_F(SchedulerTestSuite, SessionRegistryOpenDuplicateAndCloseBeforeFirstTurn) {
    const std::int32_t available_blocks = scheduler_->AvailableLcmBlocks();

    EXPECT_FALSE(scheduler_->OpenSession(""));
    EXPECT_FALSE(scheduler_->HasSession(""));
    EXPECT_TRUE(scheduler_->OpenSession("session-a"));
    EXPECT_TRUE(scheduler_->HasSession("session-a"));
    EXPECT_FALSE(scheduler_->OpenSession("session-a"));
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->PrefillSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), available_blocks);

    EXPECT_TRUE(scheduler_->CloseSession("session-a"));
    EXPECT_FALSE(scheduler_->HasSession("session-a"));
    EXPECT_FALSE(scheduler_->CloseSession("session-a"));
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), available_blocks);
}

TEST_F(SchedulerTestSuite, SessionTurnInfoUsesTheRequestPrefillFrontier) {
    EXPECT_FALSE(scheduler_->CanSubmitSessionTurn("missing"));
    EXPECT_FALSE(scheduler_->CanDeferSessionTurn("missing"));
    EXPECT_FALSE(scheduler_->SessionInfoForSession("missing").has_value());
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    EXPECT_TRUE(scheduler_->CanSubmitSessionTurn("session-a"));
    EXPECT_FALSE(scheduler_->CanDeferSessionTurn("session-a"));
    EXPECT_FALSE(scheduler_->SessionInfoForSession("session-a").has_value());

    RequestSpec first = MakeRequestSpec("turn-1", 1);
    first.session_id = "session-a";
    Submit(first);
    EXPECT_FALSE(scheduler_->CanSubmitSessionTurn("session-a"));
    auto info = scheduler_->SessionInfoForSession("session-a");
    ASSERT_TRUE(info.has_value());
    EXPECT_TRUE(info->turn_active);
    EXPECT_EQ(info->total_context_tokens, 2);
    EXPECT_EQ(info->newly_prefilled_tokens, 2);
    EXPECT_EQ(info->reused_context_tokens, 0);

    PlanOnce();
    SendForwardDone("turn-1", {99});
    SendFinish("turn-1");
    EXPECT_TRUE(scheduler_->CanSubmitSessionTurn("session-a"));
    info = scheduler_->SessionInfoForSession("session-a");
    ASSERT_TRUE(info.has_value());
    EXPECT_FALSE(info->turn_active);
    EXPECT_EQ(info->total_context_tokens, 2);
    EXPECT_EQ(info->newly_prefilled_tokens, 2);
    EXPECT_EQ(info->reused_context_tokens, 0);

    RequestSpec second = MakeRequestSpec("turn-2", 1, 100);
    second.session_id = "session-a";
    Submit(second);
    EXPECT_FALSE(scheduler_->CanSubmitSessionTurn("session-a"));
    info = scheduler_->SessionInfoForSession("session-a");
    ASSERT_TRUE(info.has_value());
    EXPECT_TRUE(info->turn_active);
    EXPECT_EQ(info->total_context_tokens, 5);
    EXPECT_EQ(info->reused_context_tokens, 2);
    EXPECT_EQ(info->newly_prefilled_tokens, 3);
}

class SessionOverlapTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.overlap_schedule_depth = 1;
        return cfg;
    }
};

TEST_F(SessionOverlapTestSuite, CanDeferSessionTurnOnlyWhileOverlapResultsDrain) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first = MakeRequestSpec("turn-1", 1);
    first.max_new_tokens = 8;
    first.session_id = "session-a";
    Submit(first);

    PlanOnce();
    SendForwardDone("turn-1", {99});
    PlanOnce();
    PlanOnce();
    SendForwardDone("turn-1", {100});
    SendFinish("turn-1");

    EXPECT_FALSE(scheduler_->CanSubmitSessionTurn("session-a"));
    EXPECT_TRUE(scheduler_->CanDeferSessionTurn("session-a"));

    SendForwardDone("turn-1", {101});
    EXPECT_TRUE(scheduler_->CanSubmitSessionTurn("session-a"));
    EXPECT_FALSE(scheduler_->CanDeferSessionTurn("session-a"));
}

class SessionFifoTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_batch_size = 2;
        cfg.max_scheduled_tokens = 4;
        return cfg;
    }
};

TEST_F(SessionFifoTestSuite, ResumedSessionMovesBehindAnInterleavedPrefill) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first_turn = MakeRequestSpec("turn-1", 1);
    first_turn.session_id = "session-a";
    Submit(first_turn);
    const ExecutionPlan first_plan = PlanOnce();
    const ForwardBatch* first_batch = FindForwardBatch(first_plan);
    ASSERT_NE(first_batch, nullptr);
    ASSERT_EQ(first_batch->request_ids, std::vector<std::string>{"turn-1"});
    SendForwardDone("turn-1", {50});
    SendFinish("turn-1");

    Submit(MakeRequestSpec("middle", 4, 100));
    const ExecutionPlan middle_plan = PlanOnce();
    const ForwardBatch* middle_batch = FindForwardBatch(middle_plan);
    ASSERT_NE(middle_batch, nullptr);
    ASSERT_EQ(middle_batch->request_ids, std::vector<std::string>{"middle"});
    ASSERT_EQ(middle_batch->input_lengths, std::vector<std::int32_t>{4});
    SendForwardDone("middle");

    RequestSpec resumed_turn = MakeRequestSpec("turn-2", 1, 200);
    resumed_turn.session_id = "session-a";
    Submit(resumed_turn);

    const ExecutionPlan competing_plan = PlanOnce();
    const ForwardBatch* competing_batch = FindForwardBatch(competing_plan);
    ASSERT_NE(competing_batch, nullptr);
    EXPECT_EQ(competing_batch->request_ids, std::vector<std::string>{"middle"});
}

class SessionRetractionInfoTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 6;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.max_batch_size = 4;
        cfg.max_scheduled_tokens = 2;
        return cfg;
    }
};

TEST_F(SessionRetractionInfoTestSuite, RetractionClearsSessionReuseFromTurnInfo) {
    ASSERT_EQ(scheduler_->MaxSingleRequestTokens(), 10);
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first_turn = MakeRequestSpec("turn-1", 1);
    first_turn.session_id = "session-a";
    Submit(first_turn);
    ExecutionPlan first_plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(first_plan), nullptr);
    SendForwardDone("turn-1", {50});
    SendFinish("turn-1");

    Submit(MakeRequestSpec("middle", 1, 100));
    ExecutionPlan middle_plan = PlanOnce();
    const ForwardBatch* middle_batch = FindForwardBatch(middle_plan);
    ASSERT_NE(middle_batch, nullptr);
    ASSERT_EQ(middle_batch->request_ids, std::vector<std::string>{"middle"});
    SendForwardDone("middle", {150});

    RequestSpec resumed_turn = MakeRequestSpec("turn-2", 3, 200);
    resumed_turn.session_id = "session-a";
    Submit(resumed_turn);
    auto info = scheduler_->SessionInfoForSession("session-a");
    ASSERT_TRUE(info.has_value());
    EXPECT_TRUE(info->turn_active);
    EXPECT_EQ(info->total_context_tokens, 9);
    EXPECT_EQ(info->reused_context_tokens, 2);
    EXPECT_EQ(info->newly_prefilled_tokens, 7);

    for (int chunk = 0; chunk < 2; ++chunk) {
        ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        ASSERT_EQ(batch->request_ids, std::vector<std::string>{"turn-2"});
        SendForwardDone("turn-2");
    }

    PlanOnce();
    ASSERT_EQ(scheduler_->WaitingSize(), 1u) << "the session continuation should have been retracted";
    info = scheduler_->SessionInfoForSession("session-a");
    ASSERT_TRUE(info.has_value());
    EXPECT_TRUE(info->turn_active);
    EXPECT_EQ(info->total_context_tokens, 9);
    EXPECT_EQ(info->reused_context_tokens, 0);
    EXPECT_EQ(info->newly_prefilled_tokens, info->total_context_tokens);
}

TEST_F(SchedulerTestSuite, SessionCumulativeContextAndGenerationReserveRejectOverLimit) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first_turn = MakeRequestSpec("turn-1", 1);
    first_turn.session_id = "session-a";
    Submit(first_turn);
    PlanOnce();
    SendForwardDone("turn-1", {50});
    SendFinish("turn-1");

    const std::int32_t context_tokens = scheduler_->RequestTokenSize("turn-1");
    const std::int32_t max_tokens = scheduler_->MaxSingleRequestTokens();
    ASSERT_GT(max_tokens, context_tokens + 1);

    RequestSpec over_limit = MakeRequestSpec("turn-2", 1, 200);
    over_limit.tokens.resize(static_cast<std::size_t>(max_tokens - context_tokens), 200);
    over_limit.max_new_tokens = 1;
    over_limit.session_id = "session-a";
    EXPECT_THROW(Submit(over_limit), std::invalid_argument);
    EXPECT_TRUE(scheduler_->CanSubmitSessionTurn("session-a"));
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-1"), context_tokens);
}

TEST_F(SchedulerTestSuite, SessionCumulativeContextAndGenerationReserveAcceptExactLimit) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first_turn = MakeRequestSpec("turn-1", 1);
    first_turn.session_id = "session-a";
    Submit(first_turn);
    PlanOnce();
    SendForwardDone("turn-1", {50});
    SendFinish("turn-1");

    const std::int32_t context_tokens = scheduler_->RequestTokenSize("turn-1");
    const std::int32_t max_tokens = scheduler_->MaxSingleRequestTokens();
    ASSERT_GT(max_tokens, context_tokens + 1);

    RequestSpec exact_limit = MakeRequestSpec("turn-2", 1, 200);
    exact_limit.tokens.resize(static_cast<std::size_t>(max_tokens - context_tokens - 1), 200);
    exact_limit.max_new_tokens = 1;
    exact_limit.session_id = "session-a";
    EXPECT_NO_THROW(Submit(exact_limit));
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-2"), max_tokens - 1);
    EXPECT_FALSE(scheduler_->CanSubmitSessionTurn("session-a"));
}

TEST_F(SchedulerTestSuite, SessionFinishRetainsPagesUntilClose) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    const std::int32_t active_before = scheduler_->ActiveLcmBlocks();
    RequestSpec turn = MakeRequestSpec("turn-1", 1);
    turn.session_id = "session-a";

    Submit(turn);
    PlanOnce();
    EXPECT_FALSE(scheduler_->CloseSession("session-a"));
    SendForwardDone("turn-1", {99});
    SendFinish("turn-1");

    EXPECT_TRUE(scheduler_->HasSession("session-a"));
    EXPECT_GT(scheduler_->ActiveLcmBlocks(), active_before);
    EXPECT_EQ(scheduler_->PrefillSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);

    EXPECT_TRUE(scheduler_->CloseSession("session-a"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), active_before);
    EXPECT_FALSE(scheduler_->HasSession("session-a"));
}

TEST_F(SchedulerTestSuite, SchedulerDestructionReleasesRetainedSessionResourcesBeforePools) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec turn = MakeRequestSpec("turn-1", 1);
    turn.session_id = "session-a";
    Submit(turn);
    PlanOnce();
    SendForwardDone("turn-1", {99});
    SendFinish("turn-1");

    ASSERT_TRUE(scheduler_->HasSession("session-a"));
    ASSERT_GT(scheduler_->ActiveLcmBlocks(), 0);
    scheduler_.reset();
}

TEST_F(SchedulerTestSuite, SessionWaitsForLateForwardResultBeforeClose) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec turn = MakeRequestSpec("turn-1", 1);
    turn.session_id = "session-a";
    Submit(turn);
    PlanOnce();
    SendForwardDone("turn-1", {99});
    PlanOnce();
    SendFinish("turn-1");

    RequestSpec next = MakeRequestSpec("turn-2", 1, 100);
    next.session_id = "session-a";
    EXPECT_THROW(Submit(next), std::invalid_argument);
    EXPECT_TRUE(scheduler_->CanDeferSessionTurn("session-a"));
    EXPECT_FALSE(scheduler_->CloseSession("session-a"));
    EXPECT_NO_THROW(SendForwardDone("turn-1", {100}));
    EXPECT_FALSE(scheduler_->CanDeferSessionTurn("session-a"));
    EXPECT_TRUE(scheduler_->CanSubmitSessionTurn("session-a"));
    EXPECT_TRUE(scheduler_->CloseSession("session-a"));
}

TEST_F(SessionOverlapTestSuite, LateForwardResultAfterStatelessFinishIsIgnored) {
    RequestSpec request = MakeRequestSpec("request", 1);
    request.max_new_tokens = 8;
    Submit(request);

    PlanOnce();
    SendForwardDone("request", {99});
    PlanOnce();
    SendFinish("request");

    EXPECT_NO_THROW(SendForwardDone("request", {100}));
}

TEST_F(SchedulerTestSuite, SessionAbortWaitsForInFlightForwardBeforeReleasingPages) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec turn = MakeRequestSpec("turn-1", 1);
    turn.session_id = "session-a";
    Submit(turn);
    PlanOnce();
    const std::int32_t active_during_forward = scheduler_->ActiveLcmBlocks();
    ASSERT_GT(active_during_forward, 0);

    SendAbortEvent("turn-1");
    EXPECT_FALSE(scheduler_->HasSession("session-a"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), active_during_forward);
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-1"), static_cast<std::int32_t>(turn.tokens.size()));
    EXPECT_NO_THROW(SendForwardDone("turn-1", {99}));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-1"), static_cast<std::int32_t>(turn.tokens.size()));

    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-1"), -1);
}

TEST_F(SchedulerTestSuite, SessionRejectsUnknownAndOverlappingTurns) {
    RequestSpec unknown = MakeRequestSpec("unknown-turn", 1);
    unknown.session_id = "missing";
    EXPECT_THROW(Submit(unknown), std::invalid_argument);

    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first = MakeRequestSpec("turn-1", 1);
    first.session_id = "session-a";
    Submit(first);

    RequestSpec overlapping = MakeRequestSpec("turn-2", 1, 100);
    overlapping.session_id = "session-a";
    EXPECT_THROW(Submit(overlapping), std::invalid_argument);
    EXPECT_FALSE(scheduler_->CloseSession("session-a"));

    SendAbortEvent("turn-1");
    PlanOnce();
    EXPECT_FALSE(scheduler_->HasSession("session-a"));
}

TEST_F(SchedulerTestSuite, SessionTurnReusesRequestAndSchedulesItsSuffix) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first = MakeRequestSpec("turn-1", 1);
    first.session_id = "session-a";
    Submit(first);
    PlanOnce();
    SendForwardDone("turn-1", {99});
    SendFinish("turn-1");

    const std::int32_t active_after_first_turn = scheduler_->ActiveLcmBlocks();
    RequestSpec next = MakeRequestSpec("turn-2", 1, 100);
    next.tokens = {101, 102};
    next.session_id = "session-a";
    Submit(next);

    EXPECT_EQ(scheduler_->RequestTokenSize("turn-1"), -1);
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-2"), 5);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), active_after_first_turn);
    ASSERT_EQ(scheduler_->PrefillSize(), 1u);

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"turn-2"});
    EXPECT_GT(batch->input_lengths.front(), 0);

    SendForwardDone("turn-2", {103});
    SendFinish("turn-2");
    EXPECT_TRUE(scheduler_->CloseSession("session-a"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(SchedulerTestSuite, ClosedSessionCleanupDoesNotEraseReusedRequestId) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    RequestSpec first = MakeRequestSpec("reused-turn", 1);
    first.session_id = "session-a";
    Submit(first);
    PlanOnce();
    SendForwardDone("reused-turn", {99});
    SendFinish("reused-turn");
    ASSERT_TRUE(scheduler_->CloseSession("session-a"));

    const RequestSpec replacement = MakeRequestSpec("reused-turn", 1, 200);
    Submit(replacement);
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"reused-turn"});
    EXPECT_EQ(scheduler_->RequestTokenSize("reused-turn"), static_cast<std::int32_t>(replacement.tokens.size()));

    SendForwardDone("reused-turn", {199});
    SendFinish("reused-turn");
    EXPECT_EQ(scheduler_->PrefillSize(), 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

TEST_F(SchedulerTestSuite, SessionTurnsRemainIsolated) {
    ASSERT_TRUE(scheduler_->OpenSession("session-a"));
    ASSERT_TRUE(scheduler_->OpenSession("session-b"));
    RequestSpec first_a = MakeRequestSpec("turn-a1", 1, 100);
    RequestSpec first_b = MakeRequestSpec("turn-b1", 1, 200);
    first_a.session_id = "session-a";
    first_b.session_id = "session-b";
    Submit(std::vector<RequestSpec>{first_a, first_b});
    PlanOnce();
    SendForwardDone("turn-a1", {301});
    SendForwardDone("turn-b1", {302});
    SendFinish("turn-a1");
    SendFinish("turn-b1");

    RequestSpec next_a = MakeRequestSpec("turn-a2", 1, 400);
    next_a.session_id = "session-a";
    Submit(next_a);
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-b1"), 3);
    EXPECT_EQ(scheduler_->RequestTokenSize("turn-a1"), -1);

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"turn-a2"});
    SendForwardDone("turn-a2", {401});
    SendFinish("turn-a2");
    EXPECT_TRUE(scheduler_->CloseSession("session-a"));
    EXPECT_TRUE(scheduler_->CloseSession("session-b"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(SchedulerTestSuite, SubmitRequestsValidatesWholeBatchBeforeInsertion) {
    const RequestSpec valid = MakeRequestSpec("valid", 1);
    RequestSpec invalid = MakeRequestSpec("invalid", 1);
    invalid.max_new_tokens = -1;

    EXPECT_THROW(Submit(std::vector<RequestSpec>{valid, invalid}), std::invalid_argument);
    EXPECT_NO_THROW(Submit(valid));
}

class HybridPrefixPromotionTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 128;
        cfg.disable_l2_cache = true;
        for (std::int32_t i = 0; i < 3; ++i) {
            CacheGroupConfig state = cfg.cache_groups.front();
            state.group_id = "linear_attention_" + std::to_string(i);
            state.family = CacheGroupFamily::State;
            cfg.cache_groups.push_back(std::move(state));
        }
        return cfg;
    }

    RequestSpec MakeHybridRequest(const std::string& id, std::int32_t suffix_start) {
        RequestSpec spec = MakeRequestSpec(id, /*num_pages=*/4);
        spec.tokens.push_back(suffix_start);
        spec.tokens.push_back(suffix_start + 1);
        spec.tokens.push_back(suffix_start + 2);
        return spec;
    }
};

class HybridSessionChunkingTestSuite : public HybridPrefixPromotionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = HybridPrefixPromotionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 512;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = 512;
        }
        return cfg;
    }
};

TEST_F(HybridPrefixPromotionTestSuite, SessionTurnRetainsHybridStatePages) {
    ASSERT_TRUE(scheduler_->OpenSession("hybrid-session"));
    RequestSpec first = MakeHybridRequest("hybrid-turn-1", 500);
    first.session_id = "hybrid-session";
    Submit(first);
    PlanOnce();
    SendForwardDone("hybrid-turn-1", {900});
    SendFinish("hybrid-turn-1");

    const std::int32_t active_after_first_turn = scheduler_->ActiveLcmBlocks();
    RequestSpec next = MakeRequestSpec("hybrid-turn-2", 1, 1000);
    next.tokens = {1001};
    next.session_id = "hybrid-session";
    Submit(next);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), active_after_first_turn);

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"hybrid-turn-2"});
    EXPECT_EQ(batch->block_tables.size(), config_.cache_groups.size());

    SendForwardDone("hybrid-turn-2", {1002});
    SendFinish("hybrid-turn-2");
    EXPECT_TRUE(scheduler_->CloseSession("hybrid-session"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(HybridSessionChunkingTestSuite, SessionContinuationKeepsHybridStateSlotsAcrossPrefillChunks) {
    ASSERT_TRUE(scheduler_->OpenSession("hybrid-session"));
    RequestSpec first = MakeHybridRequest("hybrid-turn-1", 500);
    first.session_id = "hybrid-session";
    Submit(first);

    const ExecutionPlan first_plan = PlanOnce();
    const ForwardBatch* first_batch = FindForwardBatch(first_plan);
    ASSERT_NE(first_batch, nullptr);
    const std::vector<std::int32_t>& initial_state_table = first_batch->block_tables.at("linear_attention_0").front();
    ASSERT_EQ(initial_state_table.size(), 7u);
    EXPECT_EQ(std::vector<std::int32_t>(initial_state_table.begin(), initial_state_table.begin() + 4),
              (std::vector<std::int32_t>{0, 0, 0, 0}));
    for (std::size_t slot = 4; slot < initial_state_table.size(); ++slot) {
        EXPECT_NE(initial_state_table[slot], 0);
    }

    SendForwardDone("hybrid-turn-1", {900});
    SendFinish("hybrid-turn-1");

    RequestSpec next = MakeRequestSpec("hybrid-turn-2", 1, 1000);
    next.tokens.resize(65);
    for (std::size_t i = 0; i < next.tokens.size(); ++i) {
        next.tokens[i] = 10000 + static_cast<std::int32_t>(i);
    }
    next.session_id = "hybrid-session";
    Submit(next);

    const ExecutionPlan first_suffix_plan = PlanOnce();
    const ForwardBatch* first_suffix_batch = FindForwardBatch(first_suffix_plan);
    ASSERT_NE(first_suffix_batch, nullptr);
    EXPECT_EQ(first_suffix_batch->input_lengths, std::vector<std::int32_t>{1});
    const std::vector<std::int32_t>& first_suffix_state_table =
        first_suffix_batch->block_tables.at("linear_attention_0").front();
    ASSERT_EQ(first_suffix_state_table.size(), initial_state_table.size());
    EXPECT_EQ(first_suffix_state_table[4], 0);
    EXPECT_EQ(first_suffix_state_table[5], initial_state_table[5]);
    EXPECT_EQ(first_suffix_state_table[6], initial_state_table[6]);
    SendForwardDone("hybrid-turn-2");

    const ExecutionPlan middle_suffix_plan = PlanOnce();
    const ForwardBatch* middle_suffix_batch = FindForwardBatch(middle_suffix_plan);
    ASSERT_NE(middle_suffix_batch, nullptr);
    EXPECT_EQ(middle_suffix_batch->input_lengths, std::vector<std::int32_t>{64});
    const std::vector<std::int32_t>& middle_state_table =
        middle_suffix_batch->block_tables.at("linear_attention_0").front();
    ASSERT_EQ(middle_state_table.size(), 38u);
    EXPECT_EQ(std::vector<std::int32_t>(middle_state_table.begin(), middle_state_table.begin() + 5),
              (std::vector<std::int32_t>{0, 0, 0, 0, 0}));
    for (std::size_t slot = 5; slot < middle_state_table.size(); ++slot) {
        EXPECT_NE(middle_state_table[slot], 0);
    }
    SendForwardDone("hybrid-turn-2");

    const ExecutionPlan final_suffix_plan = PlanOnce();
    const ForwardBatch* final_suffix_batch = FindForwardBatch(final_suffix_plan);
    ASSERT_NE(final_suffix_batch, nullptr);
    EXPECT_EQ(final_suffix_batch->input_lengths, std::vector<std::int32_t>{1});
    const std::vector<std::int32_t>& final_state_table =
        final_suffix_batch->block_tables.at("linear_attention_0").front();
    ASSERT_EQ(final_state_table.size(), 40u);
    EXPECT_EQ(std::vector<std::int32_t>(final_state_table.begin(), final_state_table.begin() + 37),
              std::vector<std::int32_t>(37, 0));
    for (std::size_t slot = 37; slot < final_state_table.size(); ++slot) {
        EXPECT_NE(final_state_table[slot], 0);
    }
}

TEST_F(HybridPrefixPromotionTestSuite, ThirdRequestReusesPromotedStateBoundary) {
    Submit(MakeHybridRequest("seed", 100));
    const ExecutionPlan seed_plan = PlanOnce();
    const ForwardBatch* seed = FindForwardBatch(seed_plan);
    ASSERT_NE(seed, nullptr);
    EXPECT_EQ(seed->input_lengths, std::vector<std::int32_t>{11});
    SendForwardDone("seed", {900});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit(MakeHybridRequest("promote", 200));
    const ExecutionPlan promotion_plan = PlanOnce();
    const ForwardBatch* promotion = FindForwardBatch(promotion_plan);
    ASSERT_NE(promotion, nullptr);
    ASSERT_EQ(promotion->request_ids, std::vector<std::string>{"promote"});
    EXPECT_EQ(promotion->input_lengths, std::vector<std::int32_t>{8});
    const ExecutionPlan remainder_plan = PlanOnce();
    const ForwardBatch* remainder = FindForwardBatch(remainder_plan);
    ASSERT_NE(remainder, nullptr);
    EXPECT_EQ(remainder->input_lengths, std::vector<std::int32_t>{3});
    SendForwardDone("promote", {901});
    PlanOnce();
    SendFinish("promote");
    PlanOnce();

    Submit(MakeHybridRequest("reuse", 300));
    const ExecutionPlan reuse_plan = PlanOnce();
    const ForwardBatch* reuse = FindForwardBatch(reuse_plan);
    ASSERT_NE(reuse, nullptr);
    ASSERT_EQ(reuse->request_ids, std::vector<std::string>{"reuse"});
    EXPECT_EQ(reuse->extend_prefix_lens, std::vector<std::int32_t>{8});
}

TEST(SchedulerConstructionTest, ValidatesConfigBeforeBuildingPools) {
    SchedulerConfig cfg{};
    cfg.prefix_granularity = 2;
    cfg.max_scheduled_tokens = 64;
    cfg.max_batch_size = 8;
    cfg.cache_groups.push_back(CacheGroupConfig{
        .group_id = "full_attention",
        .block_granularity = cfg.prefix_granularity,
        .total_pages = 32,
    });
    // device_allocator.total_pages stays 0, so the block pool would be built
    // with a negative usable count and assert before the config diagnostic.
    try {
        Scheduler scheduler{cfg};
        FAIL() << "a device cache without usable capacity was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string{error.what()}.find("device cache"), std::string::npos) << error.what();
    }
}

}  // namespace tokenspeed::test
