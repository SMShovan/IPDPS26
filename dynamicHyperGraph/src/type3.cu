#include <iostream>
#include <vector>
#include <climits>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../include/graphGeneration.hpp"
#include "../include/utils.hpp"
#include "../include/structure.hpp"

// Local flattener (same padding/sentinel convention as project)
static std::pair<std::vector<int>, std::vector<int>> type3LocalFlatten2D(const std::vector<std::vector<int>>& vec2d) {
    auto nextMultipleOf4 = [](int x) { return ((x + 3) / 4) * 4; };
    std::vector<int> flatValues;
    std::vector<int> startOffsets(vec2d.size());
    int cursor = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        startOffsets[i] = cursor;
        int inner = static_cast<int>(vec2d[i].size());
        int padded = nextMultipleOf4(inner);
        for (int j = 0; j < padded; ++j) {
            if (j < inner) flatValues.push_back(vec2d[i][j]);
            else if (j == padded - 1) flatValues.push_back(INT_MIN);
            else flatValues.push_back(0);
            ++cursor;
        }
    }
    return {flatValues, startOffsets};
}

// Device helpers over sentinel-padded lists
__device__ int type3Deg(const int* __restrict__ flat, int loc) {
    int count = 0;
    while (true) {
        int v = flat[loc + count];
        if (v == INT_MIN || v == 0) break;
        ++count;
    }
    return count;
}

__device__ int type3Con(const int* __restrict__ flat, int locA, int locB) {
    int i = locA;
    int j = locB;
    int count = 0;
    while (true) {
        int a = flat[i];
        int b = flat[j];
        if (a == INT_MIN || a == 0 || b == INT_MIN || b == 0) break;
        if (a == b) { ++count; ++i; ++j; }
        else if (a < b) { ++i; }
        else { ++j; }
    }
    return count;
}

__device__ int type3Group3(const int* __restrict__ flat, int locA, int locB, int locC) {
    int i = locA, j = locB, k = locC;
    int count = 0;
    while (true) {
        int a = flat[i];
        int b = flat[j];
        int c = flat[k];
        if (a == INT_MIN || a == 0 || b == INT_MIN || b == 0 || c == INT_MIN || c == 0) break;
        if (a == b && b == c) { ++count; ++i; ++j; ++k; }
        else if (a <= b && a <= c) { ++i; }
        else if (b <= a && b <= c) { ++j; }
        else { ++k; }
    }
    return count;
}

__global__ void type3CountPerHyperedgeKernel(const int* __restrict__ h2vStart,
                                             const int* __restrict__ h2vFlat,
                                             int h2vFixedSize,
                                             const int* __restrict__ h2hStart,
                                             const int* __restrict__ h2hFlat,
                                             int h2hFixedSize,
                                             long long* __restrict__ outCounts,
                                             int numHyperedges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numHyperedges) return;

    long long acc = 0;

    // Iterate neighbors j of i
    int startAdjI = h2hStart[i];
    for (int posJ = startAdjI; posJ < h2hFixedSize; ++posJ) {
        int neighborJOneBased = h2hFlat[posJ];
        if (neighborJOneBased == INT_MIN || neighborJOneBased == 0) break;
        int j = neighborJOneBased - 1;
        if (j <= i) continue; // ensure i < j

        // Intersect adj(i) and adj(j) to find common k > j
        int startAdjJ = h2hStart[j];
        int p = startAdjI;  // pointer in adj(i)
        int q = startAdjJ;  // pointer in adj(j)
        while (p < h2hFixedSize && q < h2hFixedSize) {
            int vi = h2hFlat[p];
            int vj = h2hFlat[q];
            if (vi == INT_MIN || vi == 0 || vj == INT_MIN || vj == 0) break;
            if (vi == vj) {
                int k = vi - 1; // convert to 0-based
                if (k > j) {
                    // Compute contributions for triple (i, j, k)
                    int loc_i = h2vStart[i];
                    int loc_j = h2vStart[j];
                    int loc_k = h2vStart[k];
                    int A = type3Con(h2vFlat, loc_i, loc_j);
                    int B = type3Con(h2vFlat, loc_j, loc_k);
                    int C = type3Con(h2vFlat, loc_i, loc_k);
                    int G = type3Group3(h2vFlat, loc_i, loc_j, loc_k);
                    int x = A - G;
                    int y = B - G;
                    int z = C - G;
                    if (x > 0 && y > 0 && z > 0) {
                        acc += static_cast<long long>(x) * static_cast<long long>(y) * static_cast<long long>(z);
                    }
                }
                ++p; ++q;
            } else if (vi < vj) {
                ++p;
            } else {
                ++q;
            }
        }
    }

    outCounts[i] = acc;
}

static inline void type3CheckCudaLocal(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

long long type3CountMotifs(const std::vector<std::vector<int>>& hyperedgeToVertex,
                           const std::vector<std::vector<int>>& vertexToHyperedge,
                           int payloadCapacity = (1 << 20),
                           int alignment = 4) {
    const int numHyperedges = static_cast<int>(hyperedgeToVertex.size());
    if (numHyperedges <= 0) {
        return 0;
    }

    std::vector<std::vector<int>> hyperedge2hyperedge = hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);
    auto [h2vFlatValues, h2vStartOffsets] = type3LocalFlatten2D(hyperedgeToVertex);
    auto [h2hFlatAdj, h2hStartOffsets] = type3LocalFlatten2D(hyperedge2hyperedge);

    std::vector<int> keys(numHyperedges);
    for (int i = 0; i < numHyperedges; ++i) {
        keys[i] = i + 1;
    }

    CBSTOperations h2vOps("Type3_H2V", payloadCapacity, alignment);
    h2vOps.construct(keys.data(), h2vStartOffsets.data(), numHyperedges,
                     h2vFlatValues.data(), static_cast<int>(h2vFlatValues.size()));

    CBSTOperations h2hOps("Type3_H2H", payloadCapacity, alignment);
    h2hOps.construct(keys.data(), h2hStartOffsets.data(), numHyperedges,
                     h2hFlatAdj.data(), static_cast<int>(h2hFlatAdj.size()));

    const CBSTContext& ctxH2V = h2vOps.context();
    const CBSTContext& ctxH2H = h2hOps.context();

    long long* d_counts = nullptr;
    type3CheckCudaLocal(cudaMalloc(&d_counts, numHyperedges * sizeof(long long)));
    type3CheckCudaLocal(cudaMemset(d_counts, 0, numHyperedges * sizeof(long long)));

    const int block = 256;
    const int grid = (numHyperedges + block - 1) / block;
    type3CountPerHyperedgeKernel<<<grid, block>>>(
        ctxH2V.d_startOffsets,
        ctxH2V.d_flatPayload,
        ctxH2V.fixedSize,
        ctxH2H.d_startOffsets,
        ctxH2H.d_flatPayload,
        ctxH2H.fixedSize,
        d_counts,
        numHyperedges);
    type3CheckCudaLocal(cudaDeviceSynchronize());

    std::vector<long long> counts(numHyperedges);
    type3CheckCudaLocal(cudaMemcpy(counts.data(), d_counts, numHyperedges * sizeof(long long), cudaMemcpyDeviceToHost));
    type3CheckCudaLocal(cudaFree(d_counts));

    long long total = 0;
    for (long long c : counts) {
        total += c;
    }
    return total;
}

#ifndef TYPE_MOTIF_LIBRARY
int main(int argc, char* argv[]) {
    // Parameters: <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4]
    HypergraphParams params;
    if (!parseCommandLineArgs(argc, argv, params)) {
        return 1;
    }
    printHypergraphParams(params);

    // Generate host-side mappings
    auto [hyperedgeToVertex, vertexToHyperedge] = generateHypergraph(params);
    std::vector<std::vector<int>> hyperedge2hyperedge = hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);

    // Flatten H2V and H2H
    auto [h2vFlatValues, h2vStartOffsets] = type3LocalFlatten2D(hyperedgeToVertex);
    auto [h2hFlatAdj, h2hStartOffsets] = type3LocalFlatten2D(hyperedge2hyperedge);

    // Prepare keys [1..E]
    std::vector<int> keys(params.numHyperedges);
    for (int i = 0; i < params.numHyperedges; ++i) keys[i] = i + 1;

    // Build CBSTs for H2V and H2H
    CBSTOperations h2vOps("H2V", params.payloadCapacity, params.alignment);
    h2vOps.construct(keys.data(), h2vStartOffsets.data(), params.numHyperedges,
                     h2vFlatValues.data(), static_cast<int>(h2vFlatValues.size()));

    CBSTOperations h2hOps("H2H", params.payloadCapacity, params.alignment);
    h2hOps.construct(keys.data(), h2hStartOffsets.data(), params.numHyperedges,
                     h2hFlatAdj.data(), static_cast<int>(h2hFlatAdj.size()));

    const CBSTContext& ctxH2V = h2vOps.context();
    const CBSTContext& ctxH2H = h2hOps.context();

    // Allocate output array on device
    long long* d_counts = nullptr;
    type3CheckCudaLocal(cudaMalloc(&d_counts, params.numHyperedges * sizeof(long long)));
    type3CheckCudaLocal(cudaMemset(d_counts, 0, params.numHyperedges * sizeof(long long)));

    // Launch kernel
    int block = 256;
    int grid = (params.numHyperedges + block - 1) / block;
    type3CountPerHyperedgeKernel<<<grid, block>>>(
        ctxH2V.d_startOffsets,
        ctxH2V.d_flatPayload,
        ctxH2V.fixedSize,
        ctxH2H.d_startOffsets,
        ctxH2H.d_flatPayload,
        ctxH2H.fixedSize,
        d_counts,
        params.numHyperedges
    );
    type3CheckCudaLocal(cudaDeviceSynchronize());

    // Copy back and sum on host
    std::vector<long long> counts(params.numHyperedges);
    type3CheckCudaLocal(cudaMemcpy(counts.data(), d_counts, params.numHyperedges * sizeof(long long), cudaMemcpyDeviceToHost));
    long long total = 0;
    for (long long c : counts) total += c;
    std::cout << "[GPU] Type3 motif count: " << total << std::endl;

    type3CheckCudaLocal(cudaFree(d_counts));
    return 0;
}
#endif


