#include "../include/temporal_count.hpp"
#include "../include/temporal_adjacency.hpp"
#include "../include/temporal_structure.hpp"
#include "../include/structure.hpp"
#include "../kernel/motif_utils.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

static inline void checkCudaTemporal(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

// Lookup helper: find start offset (node->value) for a given 1-based index in a CBST
static inline __device__ int find_start_offset(const CBSTNode* __restrict__ nodes, int oneBasedId) {
    const CBSTNode* node = nodes;
    while (node != nullptr && node->index != oneBasedId) {
        if (node->index > oneBasedId) node = node->left; else node = node->right;
    }
    if (node == nullptr) return -1;
    return node->value;
}

static inline __device__ int deg_arr(const int* __restrict__ arr, int loc) {
    int count = 0;
    if (loc < 0) return 0;
    while (true) {
        int v = arr[loc];
        if (v == 0 || v == INT_MIN) break;
        ++count;
        ++loc;
    }
    return count;
}

static inline __device__ int con_cross(const int* __restrict__ a, int la,
                                       const int* __restrict__ b, int lb) {
    int count = 0;
    if (la < 0 || lb < 0) return 0;
    int i = la, j = lb;
    while (true) {
        int av = a[i];
        int bv = b[j];
        if (av == INT_MIN || bv == INT_MIN || av == 0 || bv == 0) break;
        if (av == bv) { ++count; ++i; ++j; }
        else if (av < bv) { ++i; }
        else { ++j; }
    }
    return count;
}

static inline __device__ int group_cross(const int* __restrict__ a, int la,
                                         const int* __restrict__ b, int lb,
                                         const int* __restrict__ c, int lc) {
    int i = la, j = lb, k = lc;
    if (la < 0 || lb < 0 || lc < 0) return 0;
    int count = 0;
    while (true) {
        int av = a[i];
        int bv = b[j];
        int cv = c[k];
        if (av == INT_MIN || bv == INT_MIN || cv == INT_MIN || av == 0 || bv == 0 || cv == 0) break;
        if (av == bv && bv == cv) { ++count; ++i; ++j; ++k; }
        else if (av <= bv && av <= cv) { ++i; }
        else if (bv <= av && bv <= cv) { ++j; }
        else { ++k; }
    }
    return count;
}

// Adjacency-driven GPU kernel: build i and k candidate neighbors of j via V2H on device
// Limits to per-thread candidate caps to avoid dynamic allocation
#ifndef TEMPORAL_CAND_LIMIT
#define TEMPORAL_CAND_LIMIT 1024
#endif

static inline __device__ bool push_unique(int* buf, int& n, int cap, int val) {
    if (val <= 0) return false;
    for (int t = 0; t < n; ++t) if (buf[t] == val) return false;
    if (n >= cap) return false;
    buf[n++] = val;
    return true;
}

__global__ void temporalStrictIncAdjKernel(
    // H2V layers
    const CBSTNode* __restrict__ olderH2VNodes,
    const int* __restrict__ olderH2VFlat,
    const CBSTNode* __restrict__ middleH2VNodes,
    const int* __restrict__ middleH2VFlat,
    const CBSTNode* __restrict__ newestH2VNodes,
    const int* __restrict__ newestH2VFlat,
    // V2H layers
    const CBSTNode* __restrict__ olderV2HNodes,
    const int* __restrict__ olderV2HFlat,
    const CBSTNode* __restrict__ newestV2HNodes,
    const int* __restrict__ newestV2HFlat,
    // counts
    int nOlder,
    int nMiddle,
    int nNewest,
    int* __restrict__ d_counts) {
    int j0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (j0 >= nMiddle) return;
    int j = j0 + 1; // 1-based middle id

    int loc_j_mid = find_start_offset(middleH2VNodes, j);
    if (loc_j_mid < 0) return;

    // Build candidate i set using Older V2H over vertices of j
    int candI[TEMPORAL_CAND_LIMIT];
    int numI = 0;
    for (int p = loc_j_mid;; ++p) {
        int v = middleH2VFlat[p];
        if (v == 0 || v == INT_MIN) break;
        int loc_v_old = find_start_offset(olderV2HNodes, v);
        if (loc_v_old < 0) continue;
        for (int q = loc_v_old;; ++q) {
            int i = olderV2HFlat[q];
            if (i == 0 || i == INT_MIN) break;
            if (i < j && i <= nOlder) push_unique(candI, numI, TEMPORAL_CAND_LIMIT, i);
        }
    }

    // Build candidate k set using Newest V2H over vertices of j
    int candK[TEMPORAL_CAND_LIMIT];
    int numK = 0;
    for (int p = loc_j_mid;; ++p) {
        int v = middleH2VFlat[p];
        if (v == 0 || v == INT_MIN) break;
        int loc_v_new = find_start_offset(newestV2HNodes, v);
        if (loc_v_new < 0) continue;
        for (int q = loc_v_new;; ++q) {
            int k = newestV2HFlat[q];
            if (k == 0 || k == INT_MIN) break;
            if (k > j && k <= nNewest) push_unique(candK, numK, TEMPORAL_CAND_LIMIT, k);
        }
    }

    // Enumerate triples from candidate sets and tally
    for (int a = 0; a < numI; ++a) {
        int i = candI[a];
        int loc_i_old = find_start_offset(olderH2VNodes, i);
        if (loc_i_old < 0) continue;
        // ensure i~j
        if (con_cross(olderH2VFlat, loc_i_old, middleH2VFlat, loc_j_mid) <= 0) continue;
        for (int b = 0; b < numK; ++b) {
            int k = candK[b];
            int loc_k_new = find_start_offset(newestH2VNodes, k);
            if (loc_k_new < 0) continue;
            int C_jk = con_cross(middleH2VFlat, loc_j_mid, newestH2VFlat, loc_k_new);
            if (C_jk <= 0) continue;
            int C_ik = con_cross(olderH2VFlat, loc_i_old, newestH2VFlat, loc_k_new);
            if (C_ik <= 0) continue;

            int deg_i = deg_arr(olderH2VFlat, loc_i_old);
            int deg_j = deg_arr(middleH2VFlat, loc_j_mid);
            int deg_k = deg_arr(newestH2VFlat, loc_k_new);
            int C_ij = con_cross(olderH2VFlat, loc_i_old, middleH2VFlat, loc_j_mid);
            int g_ijk = group_cross(olderH2VFlat, loc_i_old, middleH2VFlat, loc_j_mid, newestH2VFlat, loc_k_new);
            int idx = compute_motif_index(deg_i, deg_j, deg_k, C_ij, C_jk, C_ik, g_ijk);
            atomicAdd(&d_counts[idx], 1);
        }
    }
}

void computeTemporalMotifCountsStrictInc(const TemporalHypergraphIndex& thg,
                                         std::vector<int>& outCounts30) {
    const CBSTContext& older = thg.h2v.context(TemporalLayer::Older);
    const CBSTContext& middle = thg.h2v.context(TemporalLayer::Middle);
    const CBSTContext& newest = thg.h2v.context(TemporalLayer::Newest);

    int nOlder = older.numRecords;
    int nMiddle = middle.numRecords;
    int nNewest = newest.numRecords;
    int* d_counts = nullptr;
    checkCudaTemporal(cudaMalloc(&d_counts, 30 * sizeof(int)));
    checkCudaTemporal(cudaMemset(d_counts, 0, 30 * sizeof(int)));

    int blockSize = 128;
    int gridSize = (nMiddle + blockSize - 1) / blockSize;
    // Use adjacency-driven kernel leveraging V2H to prune candidates
    const CBSTContext& olderV2H = thg.v2h.context(TemporalLayer::Older);
    const CBSTContext& newestV2H = thg.v2h.context(TemporalLayer::Newest);
    temporalStrictIncAdjKernel<<<gridSize, blockSize>>>(
        older.d_nodes, older.d_flatPayload,
        middle.d_nodes, middle.d_flatPayload,
        newest.d_nodes, newest.d_flatPayload,
        olderV2H.d_nodes, olderV2H.d_flatPayload,
        newestV2H.d_nodes, newestV2H.d_flatPayload,
        nOlder,
        nMiddle,
        nNewest,
        d_counts);
    checkCudaTemporal(cudaDeviceSynchronize());

    outCounts30.assign(30, 0);
    checkCudaTemporal(cudaMemcpy(outCounts30.data(), d_counts, 30 * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaTemporal(cudaFree(d_counts));
}

void computeTemporalMotifCountsStrictIncDelta(const TemporalHypergraphIndex& oldWindow,
                                              const TemporalHypergraphIndex& newWindow,
                                              std::vector<int>& outDeltaCounts30) {
    // Compute counts for both windows and subtract on host (first-cut Phase 3)
    std::vector<int> oldCounts, newCounts;
    computeTemporalMotifCountsStrictInc(oldWindow, oldCounts);
    computeTemporalMotifCountsStrictInc(newWindow, newCounts);
    outDeltaCounts30.assign(30, 0);
    for (int i = 0; i < 30; ++i) outDeltaCounts30[i] = newCounts[i] - oldCounts[i];
}

static void buildKeyPayloadPrefix(const std::unordered_map<int, std::vector<int>>& m,
                                  std::vector<int>& keys,
                                  std::vector<int>& payload,
                                  std::vector<int>& prefix) {
    keys.clear(); payload.clear(); prefix.clear();
    keys.reserve(m.size());
    for (const auto& kv : m) {
        keys.push_back(kv.first);
        for (int v : kv.second) payload.push_back(v);
        prefix.push_back(prefix.empty() ? static_cast<int>(kv.second.size())
                                        : prefix.back() + static_cast<int>(kv.second.size()));
    }
}

bool applyTemporalDeltasFromFile(const std::string& path,
                                 TemporalHypergraphIndex& thg) {
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    std::unordered_map<int, std::vector<int>> toAdd, toRemove;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        char op; int h; if (!(iss >> op >> h)) continue;
        std::vector<int> vs; int v;
        while (iss >> v) vs.push_back(v);
        if (op == 'A') toAdd[h].insert(toAdd[h].end(), vs.begin(), vs.end());
        else if (op == 'R') toRemove[h].insert(toRemove[h].end(), vs.begin(), vs.end());
    }
    std::vector<int> addKeys, addPayload, addPrefix;
    std::vector<int> remKeys, remPayload, remPrefix;
    buildKeyPayloadPrefix(toAdd, addKeys, addPayload, addPrefix);
    buildKeyPayloadPrefix(toRemove, remKeys, remPayload, remPrefix);
    if (!remKeys.empty()) thg.h2v.unfill(TemporalLayer::Newest, remKeys, remPayload, remPrefix);
    if (!addKeys.empty()) thg.h2v.fill(TemporalLayer::Newest, addKeys, addPayload, addPrefix);

    // Mirror into V2H (invert maps: vertex -> list of hyperedges)
    std::unordered_map<int, std::vector<int>> invAdd, invRemove;
    for (const auto& kv : toAdd) {
        int h = kv.first;
        for (int v : kv.second) invAdd[v].push_back(h);
    }
    for (const auto& kv : toRemove) {
        int h = kv.first;
        for (int v : kv.second) invRemove[v].push_back(h);
    }
    std::vector<int> v_addKeys, v_addPayload, v_addPrefix;
    std::vector<int> v_remKeys, v_remPayload, v_remPrefix;
    buildKeyPayloadPrefix(invAdd, v_addKeys, v_addPayload, v_addPrefix);
    buildKeyPayloadPrefix(invRemove, v_remKeys, v_remPayload, v_remPrefix);
    if (!v_remKeys.empty()) thg.v2h.unfill(TemporalLayer::Newest, v_remKeys, v_remPayload, v_remPrefix);
    if (!v_addKeys.empty()) thg.v2h.fill(TemporalLayer::Newest, v_addKeys, v_addPayload, v_addPrefix);
    return true;
}


