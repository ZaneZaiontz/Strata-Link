// src/net/stage_split_test.cpp - the split search without a GPU: placements, the RAM and pin limits, the measured
// timings and their file, failed splits, and the ranking a node of a range saves.
#include "strata/net/stage_split.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace strata::pool;

static int failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

static void test_parse_split() {
    std::vector<int64_t> at;
    CHECK(parse_split("24", at) && at.size() == 1 && at[0] == 24);
    CHECK(parse_split("16,32", at) && at.size() == 2 && !parse_split("32,16", at) && !parse_split("a", at));
}

static void test_split() {
    SplitModel m;
    m.n_layers = 48;
    for (int r = 0; r < 12288; ++r) m.profile.push_back({(int32_t) (r % 48), (int32_t) (r / 48)});
    m.slot_bytes = [](int64_t) { return (int64_t) 1 << 20; };              // 1 MiB per cached expert
    m.layer_arena_bytes.assign(48, (int64_t) 1 << 30);                     // 1 GiB of experts per layer
    m.session_bytes = [](int64_t lb, int64_t le) { return (le - lb) * ((int64_t) 8 << 20); };
    NodeCap a{"desktop", (int64_t) 4 << 30, (int64_t) 40 << 30, 0.5};
    NodeCap b{"laptop", (int64_t) 3 << 30, (int64_t) 20 << 30, 0.6};
    const SplitPlan p = auto_split(m, {a, b});
    CHECK(p.ok && p.at.size() == 1);
    CHECK(p.at[0] >= 28 && p.at[0] <= 40);                  // the laptop's RAM holds at most 20 of the 48 layers
    CHECK(p.node_arena[1] <= b.ram_room && p.node_arena[0] <= a.ram_room);
    CHECK(p.held > 0 && p.mass > 0 && p.mass <= 1.0);
    // a manual split the laptop's RAM cannot hold
    const SplitPlan bad = evaluate_split(m, {a, b}, {10});
    CHECK(!bad.ok && bad.why.find("RAM") != std::string::npos);
    // two PCs whose RAM together cannot hold the model: no placement
    NodeCap tiny{"tiny", (int64_t) 4 << 30, (int64_t) 10 << 30, 0.5};
    const SplitPlan none = auto_split(m, {tiny, tiny});
    CHECK(!none.ok && !none.why.empty());
    // three nodes: every boundary rising, every range non-empty
    const SplitPlan three = auto_split(m, {a, b, b});
    CHECK(three.ok && three.at.size() == 2 && three.at[0] >= 2 && three.at[1] > three.at[0] && three.at[1] < 48);
    // more VRAM -> more cached experts predicted
    NodeCap big_b{"laptop+", (int64_t) 12 << 30, (int64_t) 20 << 30, 0.6};
    const SplitPlan more = auto_split(m, {a, big_b});
    CHECK(more.ok && more.held >= p.held);
    // the coordinator keeps layers 0 and 1
    const SplitPlan one = evaluate_split(m, {a, b}, {1});
    CHECK(!one.ok);
}

static void test_measured_split() {
    SplitModel m;
    m.n_layers = 48;
    for (int r = 0; r < 12288; ++r) m.profile.push_back({(int32_t) (r % 48), (int32_t) (r / 48)});
    m.slot_bytes = [](int64_t) { return (int64_t) 1 << 20; };
    m.layer_arena_bytes.assign(48, (int64_t) 1 << 28);
    m.session_bytes = [](int64_t lb, int64_t le) { return (le - lb) * ((int64_t) 8 << 20); };
    NodeCap a{"desktop", (int64_t) 4 << 30, (int64_t) 40 << 30, 0.5};
    NodeCap b{"laptop", (int64_t) 3 << 30, (int64_t) 40 << 30, 0.5};
    const SplitPlan base = auto_split(m, {a, b});
    CHECK(base.ok && !base.measured && base.node_pred.size() == 2);
    double sum = 0;
    for (const double x : base.node_ms) sum += x;
    CHECK(std::fabs(base.ms - (sum + m.hop_ms * 2.0)) < 1e-9);   // the per-node parts add up to the old total
    // too few windows: not used
    SplitCalib c;
    c.add({20.0, 10.0}, {10.0, 10.0}, 12.0, 3.0, 100, base.at);
    CHECK(!c.usable(2) && std::fabs(c.scale[0] - 2.0) < 1e-9 && std::fabs(c.fixed_ms - 12.0) < 1e-9);
    c.add({20.0, 10.0}, {10.0, 10.0}, 12.0, 3.0, 150, base.at);
    CHECK(c.usable(2) && !c.usable(3) && c.windows == 250);
    SplitCalib unseen = c;
    unseen.seen.clear();
    CHECK(!unseen.usable(2));   // an entry without its measured splits is not used
    // measured: the coordinator's layers cost twice the estimate, the laptop's as estimated -> the laptop gets more
    m.calib = &c;
    const SplitPlan cal = auto_split(m, {a, b});
    CHECK(cal.ok && cal.measured && cal.at[0] < base.at[0]);
    CHECK(std::fabs(cal.ms - (cal.node_ms[0] + cal.node_ms[1] + 12.0 + 3.0)) < 1e-9);
    CHECK(std::fabs(cal.node_ms[0] - 2.0 * cal.node_pred[0]) < 1e-9);
    // measured at one split, the search stays within kTrust layers of it
    SplitCalib near_c = c;
    near_c.seen = {std::to_string(base.at[0])};
    m.calib = &near_c;
    const SplitPlan close = auto_split(m, {a, b});
    CHECK(close.ok && close.measured && std::llabs(close.at[0] - base.at[0]) <= SplitCalib::kTrust);
    CHECK(near_c.near_seen({base.at[0] + 3}) && !near_c.near_seen({base.at[0] + 4}));
    SplitCalib far_c = c;
    far_c.seen = {"16,32"};                   // measured with another pool's shape: the built-in estimate
    m.calib = &far_c;
    const SplitPlan fb = auto_split(m, {a, b});
    CHECK(fb.ok && fb.at == base.at && !fb.measured);
    m.calib = &c;
    // a bad measurement (a node without its own timing) changes nothing
    SplitCalib before = c;
    c.add({0.0, 10.0}, {10.0, 10.0}, 12.0, 3.0, 500);
    CHECK(c.windows == before.windows && c.scale == before.scale);
    // the file: round trip, other pools kept, the entry replaced
    const std::string path = "strata-pool-test-calib.txt";
    std::remove(path.c_str());
    std::string err;
    SplitCalib other;
    other.add({5.0}, {10.0}, 1.0, 0.0, 300);
    CHECK(save_calib(path, "pool A", other, err) && save_calib(path, "pool B", c, err) && save_calib(path, "pool B", c, err));
    SplitCalib r;
    CHECK(load_calib(path, "pool B", r) && r.windows == c.windows && r.scale.size() == 2 &&
          std::fabs(r.scale[0] - c.scale[0]) < 1e-3 && std::fabs(r.fixed_ms - c.fixed_ms) < 1e-3);
    CHECK(load_calib(path, "pool A", r) && r.scale.size() == 1 && !load_calib(path, "pool C", r));
    SplitCalib s2;
    s2.add({20.0, 10.0}, {10.0, 10.0}, 12.0, 3.0, 300, {33});
    s2.add({20.0, 10.0}, {10.0, 10.0}, 12.0, 3.0, 300, {30});
    CHECK(save_calib(path, "pool S", s2, err) && load_calib(path, "pool S", r) && r.seen.size() == 2 &&
          r.seen[0] == "33" && r.seen[1] == "30" && r.near_seen({27}) && !r.near_seen({26}));
    std::remove(path.c_str());
}

static void test_rank_for_range() {
    // 4 layers x 3 experts; prior ranks by expert then layer: (0,0) (1,0) (2,0) (3,0) (0,1) ...
    using P = std::pair<int32_t, int32_t>;
    std::vector<P> prior;
    for (int e = 0; e < 3; ++e)
        for (int l = 0; l < 4; ++l) prior.push_back({l, e});
    // a node of layers [2, 4) learned: its resident pairs first ((3,2) (2,1)), then the rest of the whole model
    std::vector<P> learned = {{3, 2}, {2, 1}, {2, 0}, {3, 0}, {3, 1}, {2, 2}};
    for (int e = 0; e < 3; ++e)
        for (int l = 0; l < 2; ++l) learned.push_back({l, e});
    const std::vector<P> r = rank_for_range(learned, prior, 2, 4, 4, 3);
    CHECK(r.size() == 12);
    // layers 0 and 1 keep prior's places; layers 2-3 take prior's places of layers 2-3 in learned's order
    const std::vector<P> want = {{0, 0}, {1, 0}, {3, 2}, {2, 1}, {0, 1}, {1, 1}, {2, 0}, {3, 0},
                                 {0, 2}, {1, 2}, {3, 1}, {2, 2}};
    CHECK(r == want);
    // a whole-model node, or no prior: learned as it is
    CHECK(rank_for_range(learned, prior, 0, 4, 4, 3) == learned);
    CHECK(rank_for_range(learned, {}, 2, 4, 4, 3) == learned);
    // a prior that lacks pairs: they follow at the end, every pair once
    std::vector<P> short_prior(prior.begin(), prior.begin() + 6);
    const std::vector<P> r2 = rank_for_range(learned, short_prior, 2, 4, 4, 3);
    CHECK(r2.size() == 12);
    std::vector<P> sorted = r2;
    std::sort(sorted.begin(), sorted.end());
    CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
}

static void test_split_kv_and_failures() {
    SplitModel m;
    m.n_layers = 48;
    for (int r = 0; r < 12288; ++r) m.profile.push_back({(int32_t) (r % 48), (int32_t) (r / 48)});
    m.slot_bytes = [](int64_t) { return (int64_t) 1 << 20; };
    m.layer_arena_bytes.assign(48, (int64_t) 1 << 30);                     // 1 GiB of experts per layer
    m.session_bytes = [](int64_t lb, int64_t le) { return (le - lb) * ((int64_t) 8 << 20); };
    for (int64_t l = 0; l < 48; ++l) m.layer_kv.push_back(l % 4 == 3 ? 1 : 0);   // a QSA layer every 4th
    NodeCap a{"desktop", (int64_t) 4 << 30, (int64_t) 40 << 30, 0.5};
    NodeCap b{"laptop", (int64_t) 3 << 30, (int64_t) 20 << 30, 0.4};
    b.kv_host_layer = (int64_t) 1 << 28;                                   // 0.25 GiB per QSA layer per session
    // one session: [28, 48) = 20 GiB of experts + 5 x 0.25 GiB of K/V does not fit 20 GiB; [29, 48) does
    const SplitPlan one = auto_split(m, {a, b});
    CHECK(one.ok && one.at[0] >= 29 && one.node_kv[1] > 0 && one.node_arena[1] + one.node_kv[1] <= b.ram_room);
    // three sessions (--batch 2): [31, 48) = 17 + 5 x 0.75 > 20; [32, 48) = 16 + 4 x 0.75 = 19 fits
    m.kv_copies = 3;
    const SplitPlan three = auto_split(m, {a, b});
    CHECK(three.ok && three.at[0] == 32);
    const SplitPlan at28 = evaluate_split(m, {a, b}, {28});
    CHECK(!at28.ok && at28.why.find("K/V") != std::string::npos);
    // plenty of RAM but a pin limit (Windows: about half of it): the same arithmetic against that
    NodeCap pb = b;
    pb.ram_room = (int64_t) 40 << 30;
    pb.pin_room = (int64_t) 15 << 30;
    const SplitPlan pin = auto_split(m, {a, pb});
    CHECK(pin.ok && pin.node_arena[1] + pin.node_kv[1] <= pb.pin_room);
    const SplitPlan pin30 = evaluate_split(m, {a, pb}, {30});
    CHECK(!pin30.ok && pin30.why.find("pin") != std::string::npos);
    // a split the laptop failed to load: nothing that gives it as much or more
    SplitCalib f;
    f.add_failed({35});
    f.add_failed({35});
    CHECK(f.failed.size() == 1 && f.excludes({35}, 48) && f.excludes({33}, 48) && !f.excludes({36}, 48));
    m.failures = &f;
    const SplitPlan after = auto_split(m, {a, b});
    CHECK(after.ok && after.at[0] >= 36);
    // nothing left: says why
    SplitCalib all;
    all.add_failed({47});
    m.failures = &all;
    const SplitPlan none = auto_split(m, {a, b});
    CHECK(!none.ok && none.why.find("failed to load") != std::string::npos);
    m.failures = nullptr;
    // three nodes: a worker whose range contains its failed one is excluded
    SplitCalib f3;
    f3.add_failed({16, 32});
    CHECK(f3.excludes({16, 33}, 48) && f3.excludes({17, 32}, 48) && !f3.excludes({17, 33}, 48));
    // the file keeps failed splits, also in an entry without timings yet
    const std::string path = "strata-pool-test-failed.txt";
    std::remove(path.c_str());
    std::string err;
    SplitCalib r;
    CHECK(save_calib(path, "pool F", f, err) && load_calib(path, "pool F", r) && r.failed.size() == 1 &&
          r.failed[0] == "35" && !r.usable(2) && r.excludes({30}, 48));
    std::remove(path.c_str());
}

/// A window's network time from link speeds and delays: Mbit/s against bytes, bf16 rows, unknown or nonsense speeds
/// priced as a LAN, a delay given by either node, and no link facts at all meaning the old constant per crossing.
static void test_link_cost() {
    SplitModel m;
    m.hop_overhead_ms = 0;
    const NodeCap main_pc{"main", 0, 0, 0.5}, worker{"worker", 0, 0, 0.5};
    auto net = [&](const NodeCap& a, const NodeCap& b) { return evaluate_split(m, {a, b}, {30}).ms - 24.0; };
    auto near = [](double x, double want) { return std::fabs(x - want) < 1e-9; };
    const double lan_row = 51216.0 / 117500.0;
    CHECK(near(net(main_pc, worker), 2.0 * m.hop_ms));
    NodeCap a = main_pc, b = worker;
    a.up_mbit = a.down_mbit = b.up_mbit = b.down_mbit = 20;
    CHECK(near(net(a, b), 2.0 * 20.4864));
    m.window_rows = 4;
    CHECK(near(net(a, b), 8.0 * 20.4864));
    m.window_rows = 1;
    m.row_bytes = 25608;
    CHECK(near(net(a, b), 20.4864));
    m.row_bytes = 51216;
    a = main_pc;
    a.up_mbit = 20;
    b = worker;
    b.down_mbit = -5;
    CHECK(near(net(a, b), 20.4864 + lan_row));
    a.up_mbit = 940;
    CHECK(near(net(a, b), 2.0 * lan_row));
    a = main_pc;
    a.delay_ms["worker"] = 15;
    m.hop_overhead_ms = 1.2;
    CHECK(near(net(a, worker), 2.0 * (15 + 1.2 + lan_row)));
    b.delay_ms["main"] = 15;
    CHECK(near(net(main_pc, b), 2.0 * (15 + 1.2 + lan_row)));
}

int main() {
    test_parse_split();
    test_split();
    test_rank_for_range();
    test_measured_split();
    test_split_kv_and_failures();
    test_link_cost();
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("stage_split_test: all checks passed\n");
    return 0;
}
