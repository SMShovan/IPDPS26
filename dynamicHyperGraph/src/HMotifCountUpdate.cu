#include "../include/motif_update.hpp"
#include "../kernel/motif_utils.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

static inline void checkCudaLocal(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

// Kernel: per-thread frontier hyperedge; enumerates triangles i<j<k containing frontier
template<bool IsAddition>
__global__ void motifFrontierKernel(const CBSTNode* __restrict__ d_h2hNodes,
                                    const int* __restrict__ d_h2hFlat,
                                    const CBSTNode* __restrict__ d_h2vNodes,
                                    const int* __restrict__ d_h2vFlat,
                                    const int* __restrict__ d_frontierIds, // 1-based IDs
                                    int frontierSize,
                                    int fixedSize,
                                    int* __restrict__ d_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierSize) return;
    int i = d_frontierIds[tid];

    // Retrieve adjacency start for i
    const CBSTNode* node_i = d_h2hNodes;
    while (node_i != nullptr && node_i->index != i) {
        if (node_i->index > i) node_i = node_i->left; else node_i = node_i->right;
    }
    if (node_i == nullptr) return;
    int loc_i = node_i->value;

    for (int off_j = loc_i; off_j < fixedSize; ++off_j) {
        int j = d_h2hFlat[off_j];
        if (j == 0 || j == INT_MIN) break;
        if (j <= i) continue;

        // Retrieve adjacency start for j
        const CBSTNode* node_j = d_h2hNodes;
        while (node_j != nullptr && node_j->index != j) {
            if (node_j->index > j) node_j = node_j->left; else node_j = node_j->right;
        }
        if (node_j == nullptr) continue;
        int loc_j = node_j->value;

        int p = loc_i, q = loc_j;
        while (true) {
            int a_val = d_h2hFlat[p];
            int b_val = d_h2hFlat[q];
            if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) break;
            if (a_val == b_val) {
                int k = a_val;
                if (k > j) {
                    // Fetch H2V locators
                    const CBSTNode* node_a = d_h2vNodes;
                    while (node_a != nullptr && node_a->index != i) {
                        if (node_a->index > i) node_a = node_a->left; else node_a = node_a->right;
                    }
                    const CBSTNode* node_b = d_h2vNodes;
                    while (node_b != nullptr && node_b->index != j) {
                        if (node_b->index > j) node_b = node_b->left; else node_b = node_b->right;
                    }
                    const CBSTNode* node_c = d_h2vNodes;
                    while (node_c != nullptr && node_c->index != k) {
                        if (node_c->index > k) node_c = node_c->left; else node_c = node_c->right;
                    }
                    if (node_a && node_b && node_c) {
                        int loc_a = node_a->value;
                        int loc_b = node_b->value;
                        int loc_c = node_c->value;
                        int deg_a = deg((int*)d_h2vFlat, loc_a);
                        int deg_b = deg((int*)d_h2vFlat, loc_b);
                        int deg_c = deg((int*)d_h2vFlat, loc_c);
                        int C_ab = con((int*)d_h2vFlat, loc_a, loc_b);
                        int C_bc = con((int*)d_h2vFlat, loc_b, loc_c);
                        int C_ca = con((int*)d_h2vFlat, loc_c, loc_a);
                        int g_abc = group((int*)d_h2vFlat, loc_a, loc_b, loc_c);
                        int idx = compute_motif_index(deg_a, deg_b, deg_c, C_ab, C_bc, C_ca, g_abc);
                        atomicAdd(&d_counts[idx], IsAddition ? 1 : -1);
                    }
                }
                ++p; ++q;
            } else if (a_val < b_val) {
                ++p;
            } else {
                ++q;
            }
            if (p >= fixedSize || q >= fixedSize) break;
        }
    }
}

// Build anchor flags: for each frontier hyperedge f, enumerate all triangles (f,j,k)
// and set anchor i=min(f,j,k). Flags indexed by 1-based ID, length numRecords+1.
__global__ void buildAnchorFlags(const CBSTNode* __restrict__ d_h2hNodes,
                                 const int* __restrict__ d_h2hFlat,
                                 const int* __restrict__ d_frontierIds,
                                 int frontierSize,
                                 int fixedSize,
                                 int* __restrict__ d_anchorFlags) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierSize) return;
    int f = d_frontierIds[tid];
    const CBSTNode* node_f = d_h2hNodes;
    while (node_f != nullptr && node_f->index != f) {
        if (node_f->index > f) node_f = node_f->left; else node_f = node_f->right;
    }
    if (node_f == nullptr) return;
    int loc_f = node_f->value;

    // For each neighbor j of f, intersect N(f) and N(j) to produce k
    for (int off_j = loc_f; off_j < fixedSize; ++off_j) {
        int j = d_h2hFlat[off_j];
        if (j == 0 || j == INT_MIN) break;

        const CBSTNode* node_j = d_h2hNodes;
        while (node_j != nullptr && node_j->index != j) {
            if (node_j->index > j) node_j = node_j->left; else node_j = node_j->right;
        }
        if (node_j == nullptr) continue;
        int loc_j = node_j->value;

        int p = loc_f, q = loc_j;
        while (true) {
            int a_val = d_h2hFlat[p];
            int b_val = d_h2hFlat[q];
            if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) break;
            if (a_val == b_val) {
                int k = a_val;
                int i = f;
                if (j < i) i = j;
                if (k < i) i = k;
                atomicExch(&d_anchorFlags[i], 1);
                ++p; ++q;
            } else if (a_val < b_val) {
                ++p;
            } else {
                ++q;
            }
            if (p >= fixedSize || q >= fixedSize) break;
        }
    }
}

// Enumerate triangles anchored at i (i<j<k) and accumulate counts with sign.
template<bool IsAddition>
__global__ void motifAnchorKernel(const CBSTNode* __restrict__ d_h2hNodes,
                                  const int* __restrict__ d_h2hFlat,
                                  const CBSTNode* __restrict__ d_h2vNodes,
                                  const int* __restrict__ d_h2vFlat,
                                  const int* __restrict__ d_anchorIds, // 1-based IDs (i)
                                  int anchorSize,
                                  int fixedSize,
                                  int* __restrict__ d_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= anchorSize) return;
    int i = d_anchorIds[tid];

    const CBSTNode* node_i = d_h2hNodes;
    while (node_i != nullptr && node_i->index != i) {
        if (node_i->index > i) node_i = node_i->left; else node_i = node_i->right;
    }
    if (node_i == nullptr) return;
    int loc_i = node_i->value;

    for (int off_j = loc_i; off_j < fixedSize; ++off_j) {
        int j = d_h2hFlat[off_j];
        if (j == 0 || j == INT_MIN) break;
        if (j <= i) continue;

        const CBSTNode* node_j = d_h2hNodes;
        while (node_j != nullptr && node_j->index != j) {
            if (node_j->index > j) node_j = node_j->left; else node_j = node_j->right;
        }
        if (node_j == nullptr) continue;
        int loc_j = node_j->value;

        int p = loc_i, q = loc_j;
        while (true) {
            int a_val = d_h2hFlat[p];
            int b_val = d_h2hFlat[q];
            if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) break;
            if (a_val == b_val) {
                int k = a_val;
                if (k > j) {
                    const CBSTNode* node_a = d_h2vNodes;
                    while (node_a != nullptr && node_a->index != i) {
                        if (node_a->index > i) node_a = node_a->left; else node_a = node_a->right;
                    }
                    const CBSTNode* node_b = d_h2vNodes;
                    while (node_b != nullptr && node_b->index != j) {
                        if (node_b->index > j) node_b = node_b->left; else node_b = node_b->right;
                    }
                    const CBSTNode* node_c = d_h2vNodes;
                    while (node_c != nullptr && node_c->index != k) {
                        if (node_c->index > k) node_c = node_c->left; else node_c = node_c->right;
                    }
                    if (node_a && node_b && node_c) {
                        int loc_a = node_a->value;
                        int loc_b = node_b->value;
                        int loc_c = node_c->value;
                        int deg_a = deg((int*)d_h2vFlat, loc_a);
                        int deg_b = deg((int*)d_h2vFlat, loc_b);
                        int deg_c = deg((int*)d_h2vFlat, loc_c);
                        int C_ab = con((int*)d_h2vFlat, loc_a, loc_b);
                        int C_bc = con((int*)d_h2vFlat, loc_b, loc_c);
                        int C_ca = con((int*)d_h2vFlat, loc_c, loc_a);
                        int g_abc = group((int*)d_h2vFlat, loc_a, loc_b, loc_c);
                        int idx = compute_motif_index(deg_a, deg_b, deg_c, C_ab, C_bc, C_ca, g_abc);
                        atomicAdd(&d_counts[idx], IsAddition ? 1 : -1);
                    }
                }
                ++p; ++q;
            } else if (a_val < b_val) {
                ++p;
            } else {
                ++q;
            }
            if (p >= fixedSize || q >= fixedSize) break;
        }
    }
}

void computeMotifCountsDelta(const CBSTContext& oldH2vCtx,
                             const CBSTContext& oldH2hCtx,
                             const CBSTContext& newH2vCtx,
                             const CBSTContext& newH2hCtx,
                             const std::vector<int>& deletedHyperedgeIds,
                             const std::vector<int>& insertedHyperedgeIds,
                             std::vector<int>& outDeltaCounts) {
    outDeltaCounts.assign(30, 0);
    int* d_counts = nullptr;
    checkCudaLocal(cudaMalloc(&d_counts, 30 * sizeof(int)));
    checkCudaLocal(cudaMemset(d_counts, 0, 30 * sizeof(int)));

    auto runAnchoredPhase = [&](bool isAddition,
                                const CBSTContext& h2vCtx,
                                const CBSTContext& h2hCtx,
                                const std::vector<int>& frontier){
        if (frontier.empty()) return;
        int* d_frontier = nullptr;
        checkCudaLocal(cudaMalloc(&d_frontier, frontier.size() * sizeof(int)));
        checkCudaLocal(cudaMemcpy(d_frontier, frontier.data(), frontier.size() * sizeof(int), cudaMemcpyHostToDevice));

        // Build anchor flags
        int* d_anchorFlags = nullptr;
        int flagsSize = h2hCtx.numRecords + 1; // 1-based IDs
        checkCudaLocal(cudaMalloc(&d_anchorFlags, flagsSize * sizeof(int)));
        checkCudaLocal(cudaMemset(d_anchorFlags, 0, flagsSize * sizeof(int)));
        int blockSize = 256;
        int gridF = (static_cast<int>(frontier.size()) + blockSize - 1) / blockSize;
        buildAnchorFlags<<<gridF, blockSize>>>(h2hCtx.d_nodes, h2hCtx.d_flatPayload,
                                               d_frontier, static_cast<int>(frontier.size()),
                                               h2hCtx.fixedSize, d_anchorFlags);
        checkCudaLocal(cudaDeviceSynchronize());

        // Copy flags to host and collect anchors
        std::vector<int> h_flags(flagsSize);
        checkCudaLocal(cudaMemcpy(h_flags.data(), d_anchorFlags, flagsSize * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> anchors;
        anchors.reserve(frontier.size() * 4);
        for (int id = 1; id < flagsSize; ++id) {
            if (h_flags[id]) anchors.push_back(id);
        }

        if (!anchors.empty()) {
            int* d_anchors = nullptr;
            checkCudaLocal(cudaMalloc(&d_anchors, anchors.size() * sizeof(int)));
            checkCudaLocal(cudaMemcpy(d_anchors, anchors.data(), anchors.size() * sizeof(int), cudaMemcpyHostToDevice));
            int gridA = (static_cast<int>(anchors.size()) + blockSize - 1) / blockSize;
            if (isAddition) {
                motifAnchorKernel<true><<<gridA, blockSize>>>(h2hCtx.d_nodes, h2hCtx.d_flatPayload,
                                                             h2vCtx.d_nodes, h2vCtx.d_flatPayload,
                                                             d_anchors, static_cast<int>(anchors.size()),
                                                             h2hCtx.fixedSize, d_counts);
            } else {
                motifAnchorKernel<false><<<gridA, blockSize>>>(h2hCtx.d_nodes, h2hCtx.d_flatPayload,
                                                              h2vCtx.d_nodes, h2vCtx.d_flatPayload,
                                                              d_anchors, static_cast<int>(anchors.size()),
                                                              h2hCtx.fixedSize, d_counts);
            }
            checkCudaLocal(cudaDeviceSynchronize());
            checkCudaLocal(cudaFree(d_anchors));
        }
        checkCudaLocal(cudaFree(d_anchorFlags));
        checkCudaLocal(cudaFree(d_frontier));
    };

    // Phase A: subtract using old snapshot (anchor frontier from old H2H)
    runAnchoredPhase(false, oldH2vCtx, oldH2hCtx, deletedHyperedgeIds);
    // Phase B: add using new snapshot (anchor frontier from new H2H)
    runAnchoredPhase(true, newH2vCtx, newH2hCtx, insertedHyperedgeIds);

    outDeltaCounts.resize(30);
    checkCudaLocal(cudaMemcpy(outDeltaCounts.data(), d_counts, 30 * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaLocal(cudaFree(d_counts));
}


