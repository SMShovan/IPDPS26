#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <memory>
// Include Thrust headers
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
// Include our utility functions
#include "../include/utils.hpp"
#include "../include/printUtils.hpp"
#include "../include/structure.hpp"
#include "../include/graphGeneration.hpp"
#include "../include/motif.hpp"
#include "../include/motif_update.hpp"

// Device helpers and moved kernels are provided via kernel headers
#include "../kernel/device_utils.cuh"
#include "../kernel/kernels.cuh"
#include "../kernel/motif_utils.cuh"
std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::vector<int> vec1d;
    std::vector<int> vec2dto1d(vec2d.size());

    int index = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        vec2dto1d[i] = index;
        int innerSize = vec2d[i].size();
        int paddedSize = nextMultipleOf4(innerSize);
        for (int j = 0; j < paddedSize; ++j) {
            if (j < innerSize) {
                vec1d.push_back(vec2d[i][j]);
            } else if (j == paddedSize - 1) {
                vec1d.push_back(INT_MIN); // Padding with negative infinity
            } else {
                vec1d.push_back(0); // Padding with zeros
            }
            ++index;
        }
    }

    return {vec1d, vec2dto1d};
}



void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}




namespace {

struct HostGraphState {
    std::vector<std::vector<int>> h2v;
    std::vector<std::vector<int>> v2h;
    std::vector<std::vector<int>> h2h;

    std::vector<int> h2vFlatValues;
    std::vector<int> h2vStartOffsets;
    std::vector<int> v2hFlatValues;
    std::vector<int> v2hStartOffsets;
    std::vector<int> h2hFlatValues;
    std::vector<int> h2hStartOffsets;
};

struct PreparedCBSTData {
    int* startOffsets = nullptr;
    std::unique_ptr<int[]> keys;
};

struct DeviceGraphState {
    CBSTOperations h2vOps;
    CBSTOperations v2hOps;
    CBSTOperations h2hOps;

    DeviceGraphState(const char* h2vName, const char* v2hName, const char* h2hName, int payloadCapacity, int alignment)
        : h2vOps(h2vName, payloadCapacity, alignment),
          v2hOps(v2hName, payloadCapacity, alignment),
          h2hOps(h2hName, payloadCapacity, alignment) {}
};

struct UpdateBatch {
    std::vector<int> deletedIds;
    std::vector<int> insertAssignedIds;
    std::vector<std::vector<int>> generatedInserts;

    std::vector<int> v2hRemoveKeys;
    std::vector<int> v2hRemoveValues;
    std::vector<int> v2hRemovePrefix;

    std::vector<int> v2hInsertKeys;
    std::vector<int> v2hInsertValues;
    std::vector<int> v2hInsertPrefix;

    std::vector<int> h2vInsertKeys;
    std::vector<int> h2vInsertPayload;
    std::vector<int> h2vInsertPrefix;
};

PreparedCBSTData prepareSingleCBSTInput(const std::vector<int>& startOffsets) {
    auto [preparedStartOffsets, preparedKeys] = prepareCBSTData(startOffsets);
    PreparedCBSTData data;
    data.startOffsets = preparedStartOffsets;
    data.keys.reset(preparedKeys);
    return data;
}

HostGraphState buildHostGraphState(const std::vector<std::vector<int>>& h2v, const std::vector<std::vector<int>>& v2h,
                                   const std::string& labelPrefix) {
    HostGraphState state;
    state.h2v = h2v;
    state.v2h = v2h;
    state.h2h = hyperedgeAdjacency(state.v2h, state.h2v);

    auto [h2vFlat, h2vStarts] = flatten(state.h2v, labelPrefix + "Hyperedge to Vertex");
    auto [v2hFlat, v2hStarts] = flatten(state.v2h, labelPrefix + "Vertex to Hyperedge");
    auto [h2hFlat, h2hStarts] = flatten(state.h2h, labelPrefix + "Hyperedge to Hyperedge");

    state.h2vFlatValues = std::move(h2vFlat);
    state.h2vStartOffsets = std::move(h2vStarts);
    state.v2hFlatValues = std::move(v2hFlat);
    state.v2hStartOffsets = std::move(v2hStarts);
    state.h2hFlatValues = std::move(h2hFlat);
    state.h2hStartOffsets = std::move(h2hStarts);

    return state;
}

DeviceGraphState constructDeviceGraphState(const HostGraphState& hostState, const HypergraphParams& params, int h2vNumRecords, int h2hNumRecords,
                                           const char* h2vName, const char* v2hName, const char* h2hName) {
    auto h2vPrepared = prepareSingleCBSTInput(hostState.h2vStartOffsets);
    auto v2hPrepared = prepareSingleCBSTInput(hostState.v2hStartOffsets);
    auto h2hPrepared = prepareSingleCBSTInput(hostState.h2hStartOffsets);

    DeviceGraphState deviceState(h2vName, v2hName, h2hName, params.payloadCapacity, params.alignment);
    deviceState.h2vOps.construct(h2vPrepared.keys.get(), h2vPrepared.startOffsets, h2vNumRecords,
                                 const_cast<int*>(hostState.h2vFlatValues.data()), static_cast<int>(hostState.h2vFlatValues.size()));
    deviceState.v2hOps.construct(v2hPrepared.keys.get(), v2hPrepared.startOffsets, static_cast<int>(hostState.v2h.size()),
                                 const_cast<int*>(hostState.v2hFlatValues.data()), static_cast<int>(hostState.v2hFlatValues.size()));
    deviceState.h2hOps.construct(h2hPrepared.keys.get(), h2hPrepared.startOffsets, h2hNumRecords,
                                 const_cast<int*>(hostState.h2hFlatValues.data()), static_cast<int>(hostState.h2hFlatValues.size()));

    return deviceState;
}

UpdateBatch buildSyntheticUpdateBatch(const HypergraphParams& params, const std::vector<std::vector<int>>& originalH2V) {
    UpdateBatch batch;
    int numHyperedges = params.numHyperedges;
    int numDeletes = static_cast<int>(std::llround(static_cast<double>(params.totalChanges) * (params.deletionPercentage / 100.0)));
    int numInserts = static_cast<int>(std::llround(static_cast<double>(params.totalChanges) * (params.insertionPercentage / 100.0)));

    int totalFromPercentages = numDeletes + numInserts;
    if (totalFromPercentages != params.totalChanges) {
        int diff = params.totalChanges - totalFromPercentages;
        if (params.insertionPercentage >= params.deletionPercentage) {
            numInserts += diff;
        } else {
            numDeletes += diff;
        }
    }
    if (numDeletes < 0) numDeletes = 0;
    if (numInserts < 0) numInserts = 0;

    if (numDeletes > numHyperedges) {
        std::cout << "Requested deletions exceed existing hyperedges; capping deletions from "
                  << numDeletes << " to " << numHyperedges << std::endl;
        numDeletes = numHyperedges;
        numInserts = std::max(0, params.totalChanges - numDeletes);
    }
    if (numDeletes == 0 && numInserts == 0 && params.totalChanges > 0) {
        numInserts = params.totalChanges;
    }

    std::cout << "Synthetic update batch sizes -> deletes: " << numDeletes
              << ", inserts: " << numInserts
              << " (totalChanges=" << params.totalChanges << ")" << std::endl;

    for (int k = 0; k < numDeletes; ++k) batch.deletedIds.push_back(numHyperedges - k);
    std::sort(batch.deletedIds.begin(), batch.deletedIds.end());

    batch.generatedInserts = hyperedge2vertex(numInserts, params.maxVerticesPerHyperedge, params.minVertexId, params.maxVertexId);

    int reuseCount = std::min(numDeletes, numInserts);
    batch.insertAssignedIds.resize(numInserts);
    for (int i = 0; i < reuseCount; ++i) batch.insertAssignedIds[i] = batch.deletedIds[i];
    for (int i = reuseCount; i < numInserts; ++i) batch.insertAssignedIds[i] = numHyperedges + (i - reuseCount) + 1;

    std::unordered_map<int, std::vector<int>> v2hRemovals;
    for (int hId : batch.deletedIds) {
        if (hId >= 1 && hId <= static_cast<int>(originalH2V.size())) {
            for (int vertexId : originalH2V[hId - 1]) v2hRemovals[vertexId].push_back(hId);
        }
    }
    batch.v2hRemoveKeys.reserve(v2hRemovals.size());
    for (auto& kv : v2hRemovals) {
        batch.v2hRemoveKeys.push_back(kv.first);
        for (int hId : kv.second) batch.v2hRemoveValues.push_back(hId);
        int newSize = (batch.v2hRemovePrefix.empty() ? 0 : batch.v2hRemovePrefix.back()) + static_cast<int>(kv.second.size());
        batch.v2hRemovePrefix.push_back(newSize);
    }

    std::unordered_map<int, std::vector<int>> v2hInsertions;
    for (size_t i = 0; i < batch.generatedInserts.size(); ++i) {
        int hId = batch.insertAssignedIds[i];
        for (int vertexId : batch.generatedInserts[i]) v2hInsertions[vertexId].push_back(hId);
    }
    batch.v2hInsertKeys.reserve(v2hInsertions.size());
    for (auto& kv : v2hInsertions) {
        batch.v2hInsertKeys.push_back(kv.first);
        for (int hId : kv.second) batch.v2hInsertValues.push_back(hId);
        int newSize = (batch.v2hInsertPrefix.empty() ? 0 : batch.v2hInsertPrefix.back()) + static_cast<int>(kv.second.size());
        batch.v2hInsertPrefix.push_back(newSize);
    }

    batch.h2vInsertKeys = batch.insertAssignedIds;
    for (size_t i = 0; i < batch.generatedInserts.size(); ++i) {
        for (int vertexId : batch.generatedInserts[i]) batch.h2vInsertPayload.push_back(vertexId);
        int newSize = (batch.h2vInsertPrefix.empty() ? 0 : batch.h2vInsertPrefix.back()) + static_cast<int>(batch.generatedInserts[i].size());
        batch.h2vInsertPrefix.push_back(newSize);
    }

    return batch;
}

void applyIncrementalUpdates(DeviceGraphState& deviceState, const UpdateBatch& updateBatch) {
    deviceState.h2vOps.erase(updateBatch.deletedIds);
    if (!updateBatch.v2hRemoveKeys.empty()) {
        unfillCBST(updateBatch.v2hRemoveKeys, updateBatch.v2hRemoveValues, updateBatch.v2hRemovePrefix,
                   const_cast<CBSTContext&>(deviceState.v2hOps.context()));
    }

    deviceState.h2vOps.insert(updateBatch.h2vInsertKeys, updateBatch.h2vInsertPayload, updateBatch.h2vInsertPrefix);
    if (!updateBatch.v2hInsertKeys.empty()) {
        fillCBST(updateBatch.v2hInsertKeys, updateBatch.v2hInsertValues, updateBatch.v2hInsertPrefix,
                 const_cast<CBSTContext&>(deviceState.v2hOps.context()));
    }
}

std::vector<std::vector<int>> buildUpdatedH2V(const std::vector<std::vector<int>>& originalH2V, const UpdateBatch& updateBatch) {
    int originalSize = static_cast<int>(originalH2V.size());
    int maxAssignedId = updateBatch.insertAssignedIds.empty() ? originalSize
                                                               : *std::max_element(updateBatch.insertAssignedIds.begin(), updateBatch.insertAssignedIds.end());
    int maxId = std::max(originalSize, maxAssignedId);

    std::vector<std::vector<int>> updatedH2V = originalH2V;
    if (static_cast<int>(updatedH2V.size()) < maxId) updatedH2V.resize(maxId);

    for (int hId : updateBatch.deletedIds) {
        if (hId >= 1 && hId <= static_cast<int>(updatedH2V.size())) updatedH2V[hId - 1].clear();
    }

    for (size_t i = 0; i < updateBatch.generatedInserts.size(); ++i) {
        int hId = updateBatch.insertAssignedIds[i];
        if (hId >= 1) {
            if (hId > static_cast<int>(updatedH2V.size())) updatedH2V.resize(hId);
            updatedH2V[hId - 1] = updateBatch.generatedInserts[i];
        }
    }

    return updatedH2V;
}

} // namespace

/**
 * @brief Entry point for dynamic hypergraph motif analysis.
 *
 * The program builds initial H2V/V2H/H2H structures, computes baseline motif
 * counts, applies a synthetic delete/insert batch, and then reports 30-bin
 * motif delta counts between old and updated graph frontiers.
 *
 * CLI modes:
 * - Generate mode:
 *   ./build/main <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4] [-totalChanges N] [-insertionPercentage P] [-deletionPercentage P] [--save-generated[=FILE]]
 * - Input mode:
 *   ./build/main <payload_capacity> [alignment=4] --input=FILE [-totalChanges N] [-insertionPercentage P] [-deletionPercentage P]
 *
 * In input mode, numHyperedges/maxVerticesPerHyperedge/minVertexId/maxVertexId
 * are inferred from input/FILE.
 *
 * Example:
 * @code
 * ./build/main 20 5 1 200 16384 8
 * ./build/main 8192 --input=sample.txt
 * @endcode
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument vector.
 * @return int Returns 0 on success, non-zero on argument parse failure.
 */
int main(int argc, char* argv[]) {
    // Parse command line arguments
    HypergraphParams params;
    if (!parseCommandLineArgs(argc, argv, params)) {
        return 1;
    }
    
    // Print parameters
    printHypergraphParams(params);
    
    // Build initial host-side graph state from either generation or input file.
    auto [hyperedgeToVertex, vertexToHyperedge] = loadOrGenerateHypergraph(params);
    if (hyperedgeToVertex.empty() || vertexToHyperedge.empty()) {
        std::cerr << "Failed to acquire hypergraph data." << std::endl;
        return 1;
    }
    std::vector<std::vector<int>> hyperedge2hyperedge = hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);
    std::cout << "Hyperedge to hyperedge" << std::endl;
    print2DVector(hyperedge2hyperedge);

    HostGraphState initialHostState = buildHostGraphState(hyperedgeToVertex, vertexToHyperedge, "");
    DeviceGraphState initialDeviceState = constructDeviceGraphState(initialHostState, params, params.numHyperedges, params.numHyperedges,
                                                                    "H2V", "V2H", "H2H");

    // Baseline motif counts
    computeMotifCounts(initialDeviceState.h2vOps.context(), initialDeviceState.v2hOps.context(),
                       initialDeviceState.h2hOps.context(), params.numHyperedges);

    // Build a synthetic update batch and apply incremental updates in-place
    UpdateBatch updateBatch = buildSyntheticUpdateBatch(params, initialHostState.h2v);
    applyIncrementalUpdates(initialDeviceState, updateBatch);

    // Host updated structures for rebuild H2H
    std::vector<std::vector<int>> updatedH2V = buildUpdatedH2V(initialHostState.h2v, updateBatch);
    std::vector<std::vector<int>> updatedV2H = vertex2hyperedge(updatedH2V);
    HostGraphState updatedHostState = buildHostGraphState(updatedH2V, updatedV2H, "Updated ");
    int maxId = static_cast<int>(updatedHostState.h2v.size());
    DeviceGraphState updatedDeviceState = constructDeviceGraphState(updatedHostState, params, maxId, maxId,
                                                                    "H2V-new", "V2H-new", "H2H-new");

    // --------------------------
    // CountUpdate(): subtract on deleted frontier (old), add on inserted frontier (new)
    // --------------------------
    std::vector<int> deltaCounts;
    computeMotifCountsDelta(initialDeviceState.h2vOps.context(), initialDeviceState.h2hOps.context(),
                            updatedDeviceState.h2vOps.context(), updatedDeviceState.h2hOps.context(),
                            updateBatch.deletedIds, updateBatch.insertAssignedIds, deltaCounts);
    std::cout << "Motif delta counts (30 bins): ";
    for (int i = 0; i < 30; ++i) std::cout << deltaCounts[i] << (i + 1 < 30 ? ' ' : '\n');
    
    return 0;
}