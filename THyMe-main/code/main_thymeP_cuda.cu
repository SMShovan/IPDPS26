#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "read_data.cpp"
#include "motif_id.cpp"

using namespace std;

class hmotif {
  public:
    int index_a, index_b, index_c;
    int size_a, size_b, size_c, C_ab, C_bc, C_ca, g_abc;
    hmotif(int index_a, int index_b, int index_c,
           int size_a, int size_b, int size_c, int C_ab, int C_bc, int C_ca, int g_abc) {
        this->index_a = index_a;
        this->index_b = index_b;
        this->index_c = index_c;
        this->size_a = size_a;
        this->size_b = size_b;
        this->size_c = size_c;
        this->C_ab = C_ab;
        this->C_bc = C_bc;
        this->C_ca = C_ca;
        this->g_abc = g_abc;
    }
};

inline long long convert_id(int x, int y) {
    return x * (1LL << 31) + y;
}

string hyperedge2hash(vector<int>& nodes) {
    stringstream ss;
    sort(nodes.begin(), nodes.end());
    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
        ss << nodes[i];
        if (i < static_cast<int>(nodes.size()) - 1) ss << ",";
    }
    return ss.str();
}

void update_add(int e, double time_e,
                vector<vector<int>>& V2E, vector<vector<int>>& E2V, vector<vector<double>>& E2times,
                vector<vector<pair<int, int>>>& E_adj, vector<unordered_map<int, int>>& E_inter) {
    for (const int& v : E2V[e]) {
        if (find(V2E[v].begin(), V2E[v].end(), e) == V2E[v].end()) {
            V2E[v].push_back(e);
        }
    }
    E2times[e].push_back(time_e);

    vector<long long> upd_time(E2V.size(), -1LL);
    if (E2times[e].size() == 1) {
        for (const int& v : E2V[e]) {
            for (const int& _e : V2E[v]) {
                if (e == _e) continue;
                if ((upd_time[_e] >> 31) ^ e) {
                    upd_time[_e] = ((long long)e << 31) + (int)E_adj[e].size();
                    E_adj[e].push_back({_e, 0});
                }
                E_adj[e][(int)(upd_time[_e] & 0x7FFFFFFFLL)].second++;
            }
        }
        E_inter[e].rehash(E_adj[e].size());
        for (const pair<int, int>& p : E_adj[e]) {
            int _e = p.first, C = p.second;
            E_adj[_e].push_back({e, C});
            E_inter[e].insert({_e, C});
            E_inter[_e].insert({e, C});
        }
    }
}

void update_delete(int e, int time_e,
                   vector<vector<int>>& V2E, vector<vector<int>>& E2V, vector<vector<double>>& E2times,
                   vector<vector<pair<int, int>>>& E_adj, vector<unordered_map<int, int>>& E_inter) {
    (void)time_e;
    E2times[e].erase(E2times[e].begin());
    if (E2times[e].empty()) {
        for (const int& v : E2V[e]) {
            V2E[v].erase(remove(V2E[v].begin(), V2E[v].end(), e), V2E[v].end());
        }
        for (const pair<int, int>& p : E_adj[e]) {
            int _e = p.first;
            E_inter[_e].erase(e);
            for (int i = 0; i < static_cast<int>(E_adj[_e].size()); i++) {
                if (E_adj[_e][i].first == e) {
                    E_adj[_e].erase(E_adj[_e].begin() + i);
                    break;
                }
            }
        }
        E_adj[e].clear();
        E_inter[e].clear();
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0]
             << " <dataset> <delta> [threads_ignored] [graph_file_override] [result_file_override]" << endl;
        return 1;
    }

    // Initialize CUDA context for a CUDA baseline executable.
    cudaFree(0);

    string dataset = argv[1];
    double delta = stod(argv[2]);
    string graphFile = "../data/" + dataset + "/" + dataset + ".txt";
    string writeFile = "../results/" + dataset + "_" + to_string(delta) + "_thymeP_cuda.txt";
    if (argc >= 5) graphFile = argv[4];
    if (argc >= 6) writeFile = argv[5];

    clock_t run_start = clock();
    vector<vector<int>> node2hyperedge;
    vector<vector<int>> hyperedge2node;
    vector<unordered_set<int>> hyperedge2node_set;
    vector<double> hyperedge2time;
    read_data(graphFile, node2hyperedge, hyperedge2node, hyperedge2node_set, hyperedge2time);

    int V = static_cast<int>(node2hyperedge.size());
    int E = static_cast<int>(hyperedge2node.size());
    cout << "# of nodes:\t\t\t" << V << endl;
    cout << "# of hyperedges:\t" << E << endl;
    cout << "Reading data done:\t" << (double)(clock() - run_start) / CLOCKS_PER_SEC << " sec" << endl;
    cout << "-----------------------------------------------------------" << endl << endl;

    run_start = clock();
    unordered_map<string, int> hash2index;
    vector<int> hyperedge2index;
    vector<vector<int>> V2E;
    V2E.resize(V);
    vector<vector<int>> E2V;
    vector<unordered_set<int>> E2V_set;
    vector<vector<double>> E2times;

    for (int i = 0; i < E; i++) {
        string hash = hyperedge2hash(hyperedge2node[i]);
        if (hash2index.find(hash) == hash2index.end()) {
            hash2index[hash] = static_cast<int>(hash2index.size());
            vector<int> nodes;
            unordered_set<int> nodes_set;
            for (const int& v : hyperedge2node[i]) {
                nodes.push_back(v);
                nodes_set.insert(v);
            }
            E2V.push_back(nodes);
            E2V_set.push_back(nodes_set);
            E2times.push_back(vector<double>());
        }
        hyperedge2index.push_back(hash2index[hash]);
    }

    int E_static = static_cast<int>(E2V.size());
    cout << "# of induced hyperedges:\t" << E_static << endl;
    cout << "Induced static hypergraph preprocessing done:\t"
         << (double)(clock() - run_start) / CLOCKS_PER_SEC << " sec" << endl;
    cout << "-----------------------------------------------------------" << endl << endl;

    run_start = clock();
    int x = 0;
    vector<int> hyperedge2order(E);
    iota(hyperedge2order.begin(), hyperedge2order.end(), x++);
    sort(hyperedge2order.begin(), hyperedge2order.end(),
         [&](long long i, long long j) { return hyperedge2time[i] < hyperedge2time[j]; });
    cout << "Sorting temporal hyperedges done:\t"
         << (double)(clock() - run_start) / CLOCKS_PER_SEC << " sec" << endl;
    cout << "-----------------------------------------------------------" << endl << endl;

    vector<vector<pair<int, int>>> E_adj;
    vector<unordered_map<int, int>> E_inter;
    E_adj.resize(E_static);
    E_inter.resize(E_static);

    run_start = clock();
    int start = 0;
    vector<hmotif> instances;
    vector<long long> motif2count(96, 0);
    vector<long long> upd_time(E2V.size(), -1LL);
    vector<int> intersection(max(1, E), 0);

    long long instance_cnt = 0, valid_instance_cnt = 0;
    int bin = max(1, (int)(E / 20));

    for (int end = 0; end < E; end++) {
        if (end % bin == 0) cout << end << " / " << E << endl;

        int e_a = hyperedge2order[end];
        int index_a = hyperedge2index[e_a];
        int size_a = static_cast<int>(E2V[index_a].size());
        double time_a = hyperedge2time[e_a];

        while (hyperedge2time[hyperedge2order[start]] + delta < time_a) {
            int e_start = hyperedge2order[start];
            double time_start = hyperedge2time[e_start];
            int index_start = hyperedge2index[e_start];
            update_delete(index_start, time_start, V2E, E2V, E2times, E_adj, E_inter);
            start++;
        }

        update_add(index_a, time_a, V2E, E2V, E2times, E_adj, E_inter);

        instances.clear();
        int deg_a = static_cast<int>(E_adj[index_a].size());

        for (int i = 0; i < deg_a; i++) {
            int index_b = E_adj[index_a][i].first;
            int C_ab = E_adj[index_a][i].second;
            int size_b = static_cast<int>(E2V[index_b].size());
            int deg_b = static_cast<int>(E_adj[index_b].size());

            long long ab_id = convert_id(index_a, index_b);
            const auto& nodes = E2V_set[index_b];
            auto it_end = nodes.end();
            int cnt = 0;
            for (const int& v : E2V[index_a]) {
                if (nodes.find(v) != it_end) intersection[cnt++] = v;
            }

            for (int j = 0; j < deg_b; j++) {
                int index_c = E_adj[index_b][j].first;
                int C_bc = E_adj[index_b][j].second;
                int size_c = static_cast<int>(E2V[index_c].size());
                if (index_c == index_a || index_c == index_b) continue;
                upd_time[index_c] = ab_id;

                int C_ca = E_inter[index_a][index_c], g_abc = 0;
                const auto& c_nodes = E2V_set[index_c];
                auto c_it_end = c_nodes.end();
                for (int k = 0; k < C_ab; k++) {
                    if (c_nodes.find(intersection[k]) != c_it_end) g_abc++;
                }

                if (C_ca) {
                    if (index_b < index_c) {
                        instances.push_back(hmotif(index_a, index_b, index_c, size_a, size_b, size_c, C_ab, C_bc, C_ca, g_abc));
                    }
                } else {
                    instances.push_back(hmotif(index_a, index_b, index_c, size_a, size_b, size_c, C_ab, C_bc, C_ca, g_abc));
                }
            }

            for (int j = i + 1; j < deg_a; j++) {
                int index_c = E_adj[index_a][j].first;
                int C_ca = E_adj[index_a][j].second;
                int size_c = static_cast<int>(E2V[index_c].size());

                if (upd_time[index_c] == ab_id) continue;
                if (index_c == index_a || index_c == index_b) continue;

                int C_bc = 0, g_abc = 0;
                instances.push_back(hmotif(index_a, index_b, index_c, size_a, size_b, size_c, C_ab, C_bc, C_ca, g_abc));
            }
        }

        instance_cnt += instances.size();

        for (const hmotif& instance : instances) {
            int index_b = instance.index_b, index_c = instance.index_c;
            int size_a = instance.size_a, size_b = instance.size_b, size_c = instance.size_c;
            int C_ab = instance.C_ab, C_bc = instance.C_bc, C_ca = instance.C_ca, g_abc = instance.g_abc;

            int motif_index_bc = get_motif_index(size_a, size_b, size_c, C_ab, C_bc, C_ca, g_abc, 3, 1, 2);
            int motif_index_cb = get_motif_index(size_a, size_b, size_c, C_ab, C_bc, C_ca, g_abc, 3, 2, 1);

            int bc_cnt = 0, cb_cnt = 0;
            int b_temp_cnt = static_cast<int>(E2times[index_b].size()), c_temp_cnt = static_cast<int>(E2times[index_c].size());
            for (int i = 0; i < b_temp_cnt; i++) {
                for (int j = 0; j < c_temp_cnt; j++) {
                    if (E2times[index_b][i] < E2times[index_c][j]) {
                        bc_cnt += c_temp_cnt - j;
                        break;
                    }
                }
            }
            cb_cnt = b_temp_cnt * c_temp_cnt - bc_cnt;

            motif2count[motif_index_bc] += bc_cnt;
            motif2count[motif_index_cb] += cb_cnt;
            if (bc_cnt || cb_cnt) valid_instance_cnt++;
        }

        for (int i = 0; i < deg_a; i++) {
            int index_b = E_adj[index_a][i].first;
            int C_ab = E_adj[index_a][i].second;
            int size_b = static_cast<int>(E2V[index_b].size());

            int motif_index_ab = get_motif_index(size_a, size_b, size_a, C_ab, C_ab, size_a, C_ab, 1, 2, 3);
            int motif_index_ba = get_motif_index(size_b, size_a, size_a, C_ab, size_a, C_ab, C_ab, 1, 2, 3);
            int motif_index_bb = get_motif_index(size_b, size_b, size_a, size_b, C_ab, C_ab, C_ab, 1, 2, 3);

            int ab_cnt = 0, ba_cnt = 0, bb_cnt;
            int a_temp_cnt = static_cast<int>(E2times[index_a].size()), b_temp_cnt = static_cast<int>(E2times[index_b].size());
            for (int j = 0; j < a_temp_cnt - 1; j++) {
                for (int k = 0; k < b_temp_cnt; k++) {
                    if (E2times[index_a][j] < E2times[index_b][k]) {
                        ab_cnt += b_temp_cnt - k;
                        break;
                    }
                }
            }
            ba_cnt = (a_temp_cnt - 1) * b_temp_cnt - ab_cnt;
            bb_cnt = b_temp_cnt * (b_temp_cnt - 1) / 2;

            motif2count[motif_index_ab] += ab_cnt;
            motif2count[motif_index_ba] += ba_cnt;
            motif2count[motif_index_bb] += bb_cnt;
        }

        int a_temp_cnt = static_cast<int>(E2times[index_a].size());
        if (a_temp_cnt >= 2) motif2count[95] += (a_temp_cnt - 1) * (a_temp_cnt - 2) / 2;
    }

    cout << "# of instances (total):\t" << instance_cnt << endl;
    cout << "# of instances (valid):\t" << valid_instance_cnt << endl;

    double counting_runtime = (double)(clock() - run_start) / CLOCKS_PER_SEC;
    cout << "Counting temporal h-motifs done:\t" << counting_runtime << " sec" << endl;
    cout << "TOTAL_RUNTIME_SECONDS " << fixed << counting_runtime << endl;
    cout << "-----------------------------------------------------------" << endl << endl;

    run_start = clock();
    ofstream resultFile(writeFile.c_str());
    resultFile << fixed << counting_runtime << endl;
    for (int i = 0; i < 96; i++) resultFile << i + 1 << "\t" << motif2count[i] << endl;
    resultFile.close();
    cout << "File output done:\t" << (double)(clock() - run_start) / CLOCKS_PER_SEC << " sec" << endl;
    cout << "-----------------------------------------------------------" << endl << endl;
    return 0;
}
