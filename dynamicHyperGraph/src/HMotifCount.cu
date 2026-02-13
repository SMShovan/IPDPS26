#include "../include/motif.hpp"
#include "../kernel/motif_utils.cuh"
#include "../kernel/kernels.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

static inline void checkCudaLocal(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

// Kernel: each thread processes base hyperedge i; enumerates triangles (i,j,k) with i<j<k
__global__ void motifTriangleKernel(const CBSTNode* __restrict__ d_h2hNodes,
                                    const int* __restrict__ d_h2hFlat,
                                    const CBSTNode* __restrict__ d_h2vNodes,
                                    const int* __restrict__ d_h2vFlat,
                                    int numRecords,
                                    int fixedSize,
                                    int* __restrict__ d_counts) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x; // 0-based row id
    if (i0 >= numRecords) return;
    int i = i0 + 1; // keys and adjacency IDs are 1-based

    // Find start offsets for H2H[i]
    const CBSTNode* node_i = d_h2hNodes;
    while (node_i != nullptr && node_i->index != i) {
        if (node_i->index > i) node_i = node_i->left; else node_i = node_i->right;
    }
    if (node_i == nullptr) return;
    int loc_i = node_i->value;

    // Enumerate neighbors j>i
    for (int off_j = loc_i; off_j < fixedSize; ++off_j) {
        int j = d_h2hFlat[off_j];
        if (j == 0 || j == INT_MIN) break;
        if (j <= i) continue; // enforce i<j

        // Find start offset for H2H[j]
        const CBSTNode* node_j = d_h2hNodes;
        while (node_j != nullptr && node_j->index != j) {
            if (node_j->index > j) node_j = node_j->left; else node_j = node_j->right;
        }
        if (node_j == nullptr) continue;
        int loc_j = node_j->value;

        // Intersect adjacency lists to find k; keep k>j
        int p = loc_i, q = loc_j;
        while (true) {
            int a_val = d_h2hFlat[p];
            int b_val = d_h2hFlat[q];
            if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) break;
            if (a_val == b_val) {
                int k = a_val;
                if (k > j) { // enforce i<j<k
                    // For triangle (i,j,k), get H2V locators
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
                        atomicAdd(&d_counts[idx], 1);
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

void computeMotifCounts(const CBSTContext& h2vCtx,
                        const CBSTContext& v2hCtx,
                        const CBSTContext& h2hCtx,
                        int numHyperedges) {
    (void)v2hCtx; // currently unused by kernel; kept for future extensions
    int* d_counts = nullptr;
    checkCudaLocal(cudaMalloc(&d_counts, 30 * sizeof(int)));
    checkCudaLocal(cudaMemset(d_counts, 0, 30 * sizeof(int)));

    int blockSize = 256;
    int gridSize = (numHyperedges + blockSize - 1) / blockSize;
    motifTriangleKernel<<<gridSize, blockSize>>>(h2hCtx.d_nodes,
                                                 h2hCtx.d_flatPayload,
                                                 h2vCtx.d_nodes,
                                                 h2vCtx.d_flatPayload,
                                                 numHyperedges,
                                                 h2hCtx.fixedSize,
                                                 d_counts);
    checkCudaLocal(cudaDeviceSynchronize());

    std::vector<int> counts(30, 0);
    checkCudaLocal(cudaMemcpy(counts.data(), d_counts, 30 * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaLocal(cudaFree(d_counts));

    std::cout << "Motif counts (30 bins): ";
    for (int i = 0; i < 30; ++i) {
        std::cout << counts[i] << (i + 1 < 30 ? ' ' : '\n');
    }
}


