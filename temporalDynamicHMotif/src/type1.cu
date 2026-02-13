#include <iostream>
#include <vector>
#include <climits>
#include <cuda_runtime.h>

#include "../include/graphGeneration.hpp"
#include "../include/utils.hpp"
#include "../include/structure.hpp"

// Local helper mirroring main.cu's flatten2DVector logic with sentinel padding to multiple of 4
static std::pair<std::vector<int>, std::vector<int>> localFlatten2D(const std::vector<std::vector<int>>& vec2d) {
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

__global__ void countType1PerHyperedgeKernel(const int* __restrict__ startOffsets,
                                             const int* __restrict__ flatValues,
                                             long long* __restrict__ outCounts,
                                             int numHyperedges,
                                             int fixedSize) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= numHyperedges) return;
    int start = startOffsets[e];
    // compute degree by scanning until INT_MIN or 0
    int n = 0;
    int pos = start;
    while (pos < fixedSize) {
        int v = flatValues[pos];
        if (v == INT_MIN || v == 0) break;
        ++n; ++pos;
    }
    long long c = 0;
    if (n >= 3) {
        long long nn = n;
        c = (nn * (nn - 1) * (nn - 2)) / 6;
    }
    outCounts[e] = c;
}

static inline void checkCudaLocal(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

int main(int argc, char* argv[]) {
    // Parameters: <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4]
    HypergraphParams params;
    if (!parseCommandLineArgs(argc, argv, params)) {
        return 1;
    }
    printHypergraphParams(params);

    // Generate host-side mappings
    auto [hyperedgeToVertex, vertexToHyperedge] = generateHypergraph(params);

    // Flatten H2V locally and prepare CBST inputs
    auto [h2vFlatValues, h2vStartOffsets] = localFlatten2D(hyperedgeToVertex);

    // Prepare CBST key/start arrays
    std::vector<int> h2vKeys(params.numHyperedges);
    for (int i = 0; i < params.numHyperedges; ++i) h2vKeys[i] = i + 1;

    // Build H2V CBST to allocate device buffers and copy payload
    CBSTOperations h2vOps("H2V", params.payloadCapacity, params.alignment);
    h2vOps.construct(h2vKeys.data(), h2vStartOffsets.data(), params.numHyperedges,
                     h2vFlatValues.data(), static_cast<int>(h2vFlatValues.size()));

    const CBSTContext& ctx = h2vOps.context();

    // Allocate output array on device
    long long* d_counts = nullptr;
    checkCudaLocal(cudaMalloc(&d_counts, params.numHyperedges * sizeof(long long)));
    checkCudaLocal(cudaMemset(d_counts, 0, params.numHyperedges * sizeof(long long)));

    // Launch kernel
    int block = 256;
    int grid = (params.numHyperedges + block - 1) / block;
    countType1PerHyperedgeKernel<<<grid, block>>>(ctx.d_startOffsets, ctx.d_flatPayload, d_counts, params.numHyperedges, ctx.fixedSize);
    checkCudaLocal(cudaDeviceSynchronize());

    // Copy back and reduce on host
    std::vector<long long> counts(params.numHyperedges);
    checkCudaLocal(cudaMemcpy(counts.data(), d_counts, params.numHyperedges * sizeof(long long), cudaMemcpyDeviceToHost));
    long long total = 0;
    for (long long c : counts) total += c;
    std::cout << "[GPU] Type1 motif count (sum_e C(|e|, 3)): " << total << std::endl;

    checkCudaLocal(cudaFree(d_counts));
    return 0;
}


