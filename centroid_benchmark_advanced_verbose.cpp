// centroid_benchmark_advanced_verbose.cpp
// Усложнённый benchmark с большим числом точек и прогресс-логами в stderr.
//
// Задача:
//   paint(v) — пометить вершину v красной
//   query(v) — расстояние до ближайшей красной вершины
//
// Сравниваются:
//   1) Centroid decomposition
//   2) Naive BFS на каждый query
//   3) Sqrt + LCA: периодический multi-source BFS по всем красным (O(n)),
//      между перестройками — min(base_dist[v], min по «свежим» краскам через LCA);
//      батч ≈ sqrt(n) красок (амортизированно сопоставимо с центроидом на многих режимах)
//
// stdout = CSV
// stderr = прогресс и промежуточные выводы
//
// Запуск:
//   centroid_benchmark_advanced_verbose.exe quick > results_advanced_verbose.csv
//   centroid_benchmark_advanced_verbose.exe full  > results_advanced_verbose.csv
//   centroid_benchmark_advanced_verbose.exe presentation > deck.csv
//     (слайд 14: Prüfer-like random, n=10^3..10^5, q=10^5, ~50% paint / 50% query)

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <malloc.h>
#endif

using namespace std;

namespace {

void heap_compact_hint() {
#if defined(_WIN32)
    _heapmin();
#elif defined(__GLIBC__)
    // glibc: вернуть свободные страницы ОС (если доступно).
    extern int malloc_trim(size_t);
    malloc_trim(0);
#endif
}

size_t process_private_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
            sizeof(pmc))) {
        return static_cast<size_t>(pmc.PrivateUsage);
    }
#endif
    return 0;
}

// Прирост Private Usage процесса, пока живы структуры алгоритма.
// Если ОС/куча дали заниженный delta — оставляем оценку по полям.
size_t resolve_memory_bytes(size_t baseline, size_t estimate) {
    size_t now = process_private_bytes();
    if (baseline > 0 && now > baseline) {
        size_t delta = now - baseline;
        if (delta >= estimate / 16 || estimate < 8192) return delta;
    }
    return estimate;
}

}  // namespace

struct Timer {
    using clock = chrono::steady_clock;
    clock::time_point t0;
    Timer() : t0(clock::now()) {}
    double ms() const {
        return chrono::duration<double, milli>(clock::now() - t0).count();
    }
};

struct Query {
    int type; // 0 = paint, 1 = query
    int v;
};

struct CaseCfg {
    string tree_type;
    string workload;
    int n;
    int q;
    double paint_probability;
};

// Плотная арифметическая сетка n (шаг step) — гладкие графики без «ломаных» скачков.
vector<int> make_step_n_grid(int n_min, int n_max, int step) {
    vector<int> out;
    if (step <= 0) step = 1;
    for (int n = n_min; n <= n_max; n += step) out.push_back(n);
    if (out.empty() || out.back() != n_max) out.push_back(n_max);
    return out;
}

vector<vector<int>> make_random_tree(int n, mt19937 &rng) {
    vector<vector<int>> g(n);
    if (n == 1) return g;

    vector<int> prufer(n - 2);
    for (int &x : prufer) x = uniform_int_distribution<int>(0, n - 1)(rng);

    vector<int> degree(n, 1);
    for (int x : prufer) degree[x]++;

    priority_queue<int, vector<int>, greater<int>> leaves;
    for (int i = 0; i < n; ++i) if (degree[i] == 1) leaves.push(i);

    for (int x : prufer) {
        int leaf = leaves.top();
        leaves.pop();
        g[leaf].push_back(x);
        g[x].push_back(leaf);
        if (--degree[x] == 1) leaves.push(x);
        --degree[leaf];
    }

    int a = leaves.top(); leaves.pop();
    int b = leaves.top(); leaves.pop();
    g[a].push_back(b);
    g[b].push_back(a);
    return g;
}

vector<vector<int>> make_path_tree(int n) {
    vector<vector<int>> g(n);
    for (int i = 1; i < n; ++i) {
        g[i - 1].push_back(i);
        g[i].push_back(i - 1);
    }
    return g;
}

vector<vector<int>> make_star_tree(int n) {
    vector<vector<int>> g(n);
    for (int i = 1; i < n; ++i) {
        g[0].push_back(i);
        g[i].push_back(0);
    }
    return g;
}

vector<vector<int>> make_binary_tree(int n) {
    vector<vector<int>> g(n);
    for (int i = 1; i < n; ++i) {
        int p = (i - 1) / 2;
        g[p].push_back(i);
        g[i].push_back(p);
    }
    return g;
}

vector<vector<int>> make_broom_tree(int n) {
    vector<vector<int>> g(n);
    if (n <= 1) return g;

    int path_len = max(1, n / 2);

    for (int i = 1; i < path_len; ++i) {
        g[i - 1].push_back(i);
        g[i].push_back(i - 1);
    }

    int center = path_len - 1;
    for (int i = path_len; i < n; ++i) {
        g[center].push_back(i);
        g[i].push_back(center);
    }

    return g;
}

vector<vector<int>> make_caterpillar_tree(int n) {
    vector<vector<int>> g(n);
    if (n <= 1) return g;

    int spine = max(1, n / 3);
    int next = spine;

    for (int i = 1; i < spine; ++i) {
        g[i - 1].push_back(i);
        g[i].push_back(i - 1);
    }

    for (int i = 0; i < spine && next < n; ++i) {
        for (int k = 0; k < 2 && next < n; ++k, ++next) {
            g[i].push_back(next);
            g[next].push_back(i);
        }
    }

    while (next < n) {
        int p = next % spine;
        g[p].push_back(next);
        g[next].push_back(p);
        ++next;
    }

    return g;
}

vector<vector<int>> make_tree(const string &tree_type, int n, mt19937 &rng) {
    if (tree_type == "random") return make_random_tree(n, rng);
    if (tree_type == "path") return make_path_tree(n);
    if (tree_type == "star") return make_star_tree(n);
    if (tree_type == "binary") return make_binary_tree(n);
    if (tree_type == "broom") return make_broom_tree(n);
    if (tree_type == "caterpillar") return make_caterpillar_tree(n);
    throw runtime_error("unknown tree type: " + tree_type);
}

vector<Query> make_queries(int n, int q, double paint_probability, mt19937 &rng) {
    vector<Query> queries;
    queries.reserve(q);

    uniform_int_distribution<int> vertex_dist(0, n - 1);
    uniform_real_distribution<double> prob(0.0, 1.0);

    queries.push_back({0, vertex_dist(rng)});

    for (int i = 1; i < q; ++i) {
        int type = (prob(rng) < paint_probability) ? 0 : 1;
        queries.push_back({type, vertex_dist(rng)});
    }

    return queries;
}

class CentroidDecomposition {
public:
    explicit CentroidDecomposition(const vector<vector<int>> &graph)
        : g(graph), n((int)graph.size()), used(n, false), sub(n, 0),
          parent_tmp(n, -1), best(n, INF), chains(n), total_chain_pairs(0) {
        build(0);
        for (const auto &v : chains) total_chain_pairs += v.size();
    }

    void paint(int v) {
        for (auto [c, dist] : chains[v]) if (dist < best[c]) best[c] = dist;
    }

    int query(int v) const {
        int ans = INF;
        for (auto [c, dist] : chains[v]) ans = min(ans, best[c] + dist);
        return ans;
    }

    size_t memory_bytes_estimate() const {
        return total_chain_pairs * sizeof(pair<int, int>)
             + best.size() * sizeof(int)
             + sub.size() * sizeof(int)
             + parent_tmp.size() * sizeof(int)
             + used.size() * sizeof(bool);
    }

private:
    static constexpr int INF = 1'000'000'000;

    const vector<vector<int>> &g;
    int n;
    vector<bool> used;
    vector<int> sub;
    vector<int> parent_tmp;
    vector<int> best;
    vector<vector<pair<int, int>>> chains;
    size_t total_chain_pairs;

    vector<int> get_component_order(int entry) {
        vector<int> order;
        vector<int> st;

        st.push_back(entry);
        parent_tmp[entry] = -1;

        while (!st.empty()) {
            int v = st.back();
            st.pop_back();

            order.push_back(v);

            for (int to : g[v]) {
                if (to == parent_tmp[v] || used[to]) continue;
                parent_tmp[to] = v;
                st.push_back(to);
            }
        }

        return order;
    }

    int find_centroid_from_order(const vector<int> &order) {
        int total = (int)order.size();

        for (int v : order) sub[v] = 1;

        for (int i = (int)order.size() - 1; i >= 0; --i) {
            int v = order[i];
            for (int to : g[v]) {
                if (!used[to] && parent_tmp[to] == v) sub[v] += sub[to];
            }
        }

        int c = order[0];

        while (true) {
            bool moved = false;

            for (int to : g[c]) {
                if (used[to]) continue;

                int part;
                if (parent_tmp[to] == c) part = sub[to];
                else if (parent_tmp[c] == to) part = total - sub[c];
                else continue;

                if (part > total / 2) {
                    c = to;
                    moved = true;
                    break;
                }
            }

            if (!moved) return c;
        }
    }

    void collect_distances_iterative(int centroid) {
        vector<tuple<int, int, int>> st;
        st.push_back({centroid, -1, 0});

        while (!st.empty()) {
            auto [v, p, d] = st.back();
            st.pop_back();

            chains[v].push_back({centroid, d});

            for (int to : g[v]) {
                if (to == p || used[to]) continue;
                st.push_back({to, v, d + 1});
            }
        }
    }

    void build(int entry) {
        vector<int> order = get_component_order(entry);
        int c = find_centroid_from_order(order);

        collect_distances_iterative(c);
        used[c] = true;

        for (int to : g[c]) {
            if (!used[to]) build(to);
        }
    }
};

class NaiveBFS {
public:
    explicit NaiveBFS(const vector<vector<int>> &graph)
        : g(graph), red(graph.size(), false), dist(graph.size(), -1) {}

    void paint(int v) { red[v] = true; }

    int query(int start) {
        fill(dist.begin(), dist.end(), -1);

        queue<int> q;
        q.push(start);
        dist[start] = 0;

        while (!q.empty()) {
            int v = q.front();
            q.pop();

            if (red[v]) return dist[v];

            for (int to : g[v]) {
                if (dist[to] == -1) {
                    dist[to] = dist[v] + 1;
                    q.push(to);
                }
            }
        }

        return INF;
    }

    size_t memory_bytes_estimate() const {
        return red.size() * sizeof(bool) + dist.size() * sizeof(int);
    }

private:
    static constexpr int INF = 1'000'000'000;

    const vector<vector<int>> &g;
    vector<bool> red;
    vector<int> dist;
};

class SqrtLcaNearestRed {
public:
    explicit SqrtLcaNearestRed(const vector<vector<int>> &graph)
        : g(graph), n((int)g.size()), is_red(n, false), base_dist(n, INF) {
        if (n <= 0) {
            LOG = 1;
            batch = 1;
            return;
        }

        LOG = 1;
        while ((1 << LOG) <= n) ++LOG;

        batch = max(2, (int)sqrt((double)n));

        depth.assign(n, 0);
        up.assign(LOG, vector<int>(n, 0));

        vector<int> parent(n, -1);
        queue<int> qq;
        qq.push(0);
        parent[0] = 0;
        depth[0] = 0;
        up[0][0] = 0;

        while (!qq.empty()) {
            int v = qq.front();
            qq.pop();

            for (int to : g[v]) {
                if (to == parent[v]) continue;
                parent[to] = v;
                depth[to] = depth[v] + 1;
                up[0][to] = v;
                qq.push(to);
            }
        }

        for (int k = 1; k < LOG; ++k) {
            for (int v = 0; v < n; ++v) up[k][v] = up[k - 1][up[k - 1][v]];
        }
    }

    void paint(int v) {
        if (is_red[v]) return;
        is_red[v] = true;
        pending.push_back(v);
        if ((int)pending.size() >= batch) rebuild();
    }

    int query(int v) const {
        if (!any_red()) return INF;

        int ans = base_dist[v];
        for (int u : pending) ans = min(ans, dist(v, u));
        return ans;
    }

    size_t memory_bytes_estimate() const {
        return is_red.size() * sizeof(bool) + pending.capacity() * sizeof(int)
             + base_dist.size() * sizeof(int) + depth.size() * sizeof(int)
             + up.size() * (up.empty() ? 0 : up[0].size() * sizeof(int));
    }

private:
    static constexpr int INF = 1'000'000'000;

    const vector<vector<int>> &g;
    int n;
    int LOG = 1;
    int batch = 1;
    vector<bool> is_red;
    vector<int> pending;
    vector<int> base_dist;
    vector<int> depth;
    vector<vector<int>> up;

    bool any_red() const {
        for (bool b : is_red)
            if (b) return true;
        return false;
    }

    void rebuild() {
        fill(base_dist.begin(), base_dist.end(), INF);
        queue<int> q;

        for (int i = 0; i < n; ++i) {
            if (is_red[i]) {
                base_dist[i] = 0;
                q.push(i);
            }
        }

        while (!q.empty()) {
            int v = q.front();
            q.pop();

            for (int to : g[v]) {
                if (base_dist[to] == INF) {
                    base_dist[to] = base_dist[v] + 1;
                    q.push(to);
                }
            }
        }

        pending.clear();
    }

    int lca(int a, int b) const {
        if (depth[a] < depth[b]) swap(a, b);

        int diff = depth[a] - depth[b];
        for (int k = 0; k < LOG; ++k) {
            if (diff & (1 << k)) a = up[k][a];
        }

        if (a == b) return a;

        for (int k = LOG - 1; k >= 0; --k) {
            if (up[k][a] != up[k][b]) {
                a = up[k][a];
                b = up[k][b];
            }
        }

        return up[0][a];
    }

    int dist(int a, int b) const {
        int w = lca(a, b);
        return depth[a] + depth[b] - 2 * depth[w];
    }
};

uint64_t run_centroid(const vector<vector<int>> &g, const vector<Query> &queries,
                      double &build_ms, double &run_ms, size_t &memory_bytes) {
    heap_compact_hint();
    size_t baseline = process_private_bytes();

    uint64_t checksum = 0;
  {
    Timer build_timer;
    CentroidDecomposition cd(g);
    build_ms = build_timer.ms();

    Timer run_timer;
    for (const auto &qq : queries) {
        if (qq.type == 0) cd.paint(qq.v);
        else checksum += (uint64_t)cd.query(qq.v);
    }
    run_ms = run_timer.ms();

    memory_bytes = resolve_memory_bytes(baseline, cd.memory_bytes_estimate());
  }
    return checksum;
}

uint64_t run_naive(const vector<vector<int>> &g, const vector<Query> &queries,
                   double &run_ms, size_t &memory_bytes) {
    heap_compact_hint();
    size_t baseline = process_private_bytes();

    uint64_t checksum = 0;
  {
    NaiveBFS naive(g);

    Timer run_timer;
    for (const auto &qq : queries) {
        if (qq.type == 0) naive.paint(qq.v);
        else checksum += (uint64_t)naive.query(qq.v);
    }
    run_ms = run_timer.ms();

    memory_bytes = resolve_memory_bytes(baseline, naive.memory_bytes_estimate());
  }
    return checksum;
}

uint64_t run_lca(const vector<vector<int>> &g, const vector<Query> &queries,
                 double &build_ms, double &run_ms, size_t &memory_bytes) {
    heap_compact_hint();
    size_t baseline = process_private_bytes();

    uint64_t checksum = 0;
  {
    Timer build_timer;
    SqrtLcaNearestRed lca_ds(g);
    build_ms = build_timer.ms();

    Timer run_timer;
    for (const auto &qq : queries) {
        if (qq.type == 0) lca_ds.paint(qq.v);
        else checksum += (uint64_t)lca_ds.query(qq.v);
    }
    run_ms = run_timer.ms();

    memory_bytes = resolve_memory_bytes(baseline, lca_ds.memory_bytes_estimate());
  }
    return checksum;
}

void bench_case(const CaseCfg &cfg, int case_no, int total_cases, mt19937 &rng) {
    cerr << "[" << case_no << "/" << total_cases << "] "
         << "tree=" << cfg.tree_type
         << ", workload=" << cfg.workload
         << ", n=" << cfg.n
         << ", q=" << cfg.q
         << ", paint_prob=" << fixed << setprecision(2) << cfg.paint_probability
         << "\n";

    Timer total_timer;

    vector<vector<int>> g = make_tree(cfg.tree_type, cfg.n, rng);
    vector<Query> queries = make_queries(cfg.n, cfg.q, cfg.paint_probability, rng);

    int paint_count = 0;
    int query_count = 0;
    for (const auto &qq : queries) {
        if (qq.type == 0) ++paint_count;
        else ++query_count;
    }

    double cd_build_ms = 0.0;
    double cd_run_ms = 0.0;
    double naive_run_ms = 0.0;
    double lca_build_ms = 0.0;
    double lca_run_ms = 0.0;

    size_t cd_mem = 0;
    size_t naive_mem = 0;
    size_t lca_mem = 0;

    uint64_t cd_sum = run_centroid(g, queries, cd_build_ms, cd_run_ms, cd_mem);
    uint64_t naive_sum = run_naive(g, queries, naive_run_ms, naive_mem);
    uint64_t lca_sum = run_lca(g, queries, lca_build_ms, lca_run_ms, lca_mem);

    if (cd_sum != naive_sum || cd_sum != lca_sum) {
        cerr << "ERROR: checksum mismatch"
             << " tree=" << cfg.tree_type
             << " workload=" << cfg.workload
             << " n=" << cfg.n
             << " cd=" << cd_sum
             << " naive=" << naive_sum
             << " lca=" << lca_sum
             << "\n";
        exit(1);
    }

    double speedup_vs_naive = naive_run_ms / max(cd_run_ms, 1e-9);
    double speedup_vs_lca = lca_run_ms / max(cd_run_ms, 1e-9);
    double total_ms = total_timer.ms();

        cerr << "    ok: centroid=" << fixed << setprecision(2) << cd_run_ms << " ms"
         << ", naive=" << naive_run_ms << " ms"
         << ", sqrt_lca=" << lca_run_ms << " ms"
         << ", mem(KiB) cd/naive/lca="
         << (cd_mem / 1024) << "/" << (naive_mem / 1024) << "/" << (lca_mem / 1024)
         << ", speedup_vs_naive=" << speedup_vs_naive << "x"
         << ", total=" << total_ms << " ms"
         << "\n";

    if (speedup_vs_naive > 50.0) {
        cerr << "    вывод: на этом размере центроидная декомпозиция уже даёт крупный выигрыш.\n";
    } else if (speedup_vs_naive > 5.0) {
        cerr << "    вывод: выигрыш заметный, но ещё не экстремальный.\n";
    } else {
        cerr << "    вывод: на малом n константы ещё могут быть сравнимы с наивным подходом.\n";
    }

    cout << cfg.tree_type << ','
         << cfg.workload << ','
         << cfg.n << ','
         << cfg.q << ','
         << paint_count << ','
         << query_count << ','
         << fixed << setprecision(3)
         << cd_build_ms << ','
         << cd_run_ms << ','
         << naive_run_ms << ','
         << lca_build_ms << ','
         << lca_run_ms << ','
         << speedup_vs_naive << ','
         << speedup_vs_lca << ','
         << cd_mem << ','
         << naive_mem << ','
         << lca_mem << ','
         << cd_sum
         << '\n';
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string mode = "quick";
    if (argc >= 2) mode = argv[1];

    mt19937 rng(42);

    vector<string> tree_types;
    vector<pair<string, double>> workloads;
    vector<int> sizes;
    int q = 0;

    if (mode == "quick" || mode == "demo") {
        // Основной бенчмарк: random + path, плотная сетка n (шаг 100), q = 50k.
        // ~298 значений n x 2 дерева x 3 нагрузки ≈ 1788 кейсов на seed.
        tree_types = {"random", "path"};
        workloads = {
            {"read_heavy", 0.01},
            {"balanced", 0.15},
            {"update_heavy", 0.50}
        };
        sizes = make_step_n_grid(300, 30000, 100);
        q = 50000;
    } else if (mode == "full") {
        tree_types = {"random", "path"};
        workloads = {
            {"read_heavy", 0.01},
            {"balanced", 0.15},
            {"update_heavy", 0.50}
        };
        sizes = make_step_n_grid(300, 30000, 100);
        q = 50000;
    } else if (mode == "presentation") {
        // Слайд 14: случайные деревья (Prüfer-like), n от 10^3 до 10^5, q = 10^5,
        // «50% update / 50% query» моделируем вероятностью paint 0.5 на каждом шаге
        // (первый запрос всегда paint, как в остальных режимах).
        tree_types = {"random"};
        workloads = {
            {"deck_50_50", 0.5},
        };
        // Сетка по n: ~столько же «точек», что в quick, но в диапазоне презентации.
        sizes = {1000,  2000,  3000,  5000,  8000,  10000, 15000, 20000, 30000,
                 40000, 50000, 65000, 80000, 100000};
        q = 100000;
    } else if (mode == "smoke") {
        // Очень маленький набор: проверка чексумм и сборка, секунды.
        tree_types = {"random", "path"};
        workloads = {{"balanced", 0.15}};
        sizes = {300, 500};
        q = 600;
    } else {
        cerr << "Unknown mode: " << mode << "\n";
        cerr << "Use: presentation, smoke, demo, quick or full\n";
        return 2;
    }

    vector<CaseCfg> cases;
    for (const string &tree : tree_types) {
        for (const auto &[workload_name, paint_probability] : workloads) {
            for (int n : sizes) {
                cases.push_back({tree, workload_name, n, q, paint_probability});
            }
        }
    }

    {
        set<int> uniq_n;
        for (const auto &c : cases) uniq_n.insert(c.n);
        cerr << "Уникальных значений n: " << uniq_n.size() << " — ";
        bool fst = true;
        for (int x : uniq_n) {
            if (!fst) cerr << ", ";
            fst = false;
            cerr << x;
        }
        cerr << "\n";
    }

    cerr << "Mode: " << mode << "\n";
    if (mode == "presentation") {
        cerr << "Презентация (слайды 14–15): random Prüfer-like, n=10^3..10^5, q=10^5, p(paint)=0.5.\n";
        cerr << "Слайд 14: в фокусе Naive BFS vs Centroid; sqrt+LCA в CSV для сравнения.\n";
        cerr << "Память: прирост Private Usage процесса (Windows), между прогонами _heapmin().\n";
        cerr << "Внимание: наивный BFS на n≈10^5 может считаться очень долго.\n";
    }
    cerr << "Total benchmark cases: " << cases.size() << "\n";
    cerr << "CSV goes to stdout, progress goes to stderr.\n\n";

    cout << "tree_type,workload,n,queries,paint_count,query_count,"
         << "centroid_build_ms,centroid_run_ms,naive_bfs_run_ms,"
         << "lca_build_ms,lca_run_ms,"
         << "speedup_vs_naive,speedup_vs_lca,"
         << "centroid_memory_bytes,naive_memory_bytes,lca_memory_bytes,"
         << "checksum\n";

    Timer global_timer;

    for (int i = 0; i < (int)cases.size(); ++i) {
        bench_case(cases[i], i + 1, (int)cases.size(), rng);
    }

    cerr << "\nAll done in " << fixed << setprecision(2) << global_timer.ms() / 1000.0 << " seconds.\n";
    cerr << "Главный ожидаемый вывод: при росте n центроидная декомпозиция масштабируется заметно лучше,\n";
    cerr << "потому что каждый запрос обрабатывается по цепочке центроидов длины O(log n), а не обходом дерева.\n";

    return 0;
}
