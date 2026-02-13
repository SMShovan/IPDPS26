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
#include "../include/temporal_structure.hpp"
#include "../include/temporal_adjacency.hpp"
#include "../include/temporal_count.hpp"

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


// ceil_log2 and floor_log2 are provided by kernel/device_utils.cuh

// CBSTNode and CBSTContext are declared in structure.hpp

// Kernels included from kernel/*.cu via kernels.cuh

// constructCBST moved to structure/operations.cu

// insertCBST moved to structure/operations.cu

// deleteCBST moved to structure/operations.cu





// Moved delete/avail kernels are included via kernels.cuh




// Payload kernels included via kernels.cuh

// operations moved to structure/operations.cu


int main(int argc, char* argv[]) {
    // Parse command line arguments
    HypergraphParams params;
    if (!parseCommandLineArgs(argc, argv, params)) {
        return 1;
    }
    
    // Print parameters
    printHypergraphParams(params);
    
    // Generate hypergraph mappings
    auto [hyperedgeToVertex, vertexToHyperedge] = generateHypergraph(params);
    
    // Generate hyperedge-to-hyperedge adjacency
    std::vector<std::vector<int>> hyperedge2hyperedge = hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);
    std::cout << "Hyperedge to hyperedge" << std::endl;
    print2DVector(hyperedge2hyperedge);

    // Flatten the 2D vectors for GPU processing
    auto [h2vFlatVertexIds, h2vStartOffsets] = flatten(hyperedgeToVertex, "Hyperedge to Vertex");
    auto [v2hFlatHyperedgeIds, v2hStartOffsets] = flatten(vertexToHyperedge, "Vertex to Hyperedge");
    auto [h2hFlatAdjacency, h2hStartOffsets] = flatten(hyperedge2hyperedge, "Hyperedge to Hyperedge");

    // Prepare data for Complete Binary Search Tree construction
    auto [cbstH2VStartOffsets, cbstH2VKeys] = prepareCBSTData(h2vStartOffsets);
    auto [cbstV2HStartOffsets, cbstV2HKeys] = prepareCBSTData(v2hStartOffsets);
    auto [cbstH2HStartOffsets, cbstH2HKeys] = prepareCBSTData(h2hStartOffsets);

    // Construct Complete Binary Search Trees (generic) for each dataset using OO wrapper
    int numVertices = static_cast<int>(vertexToHyperedge.size());

    CBSTOperations h2vOps("H2V", params.payloadCapacity, params.alignment);
    h2vOps.construct(cbstH2VKeys, cbstH2VStartOffsets, params.numHyperedges,
                     h2vFlatVertexIds.data(), static_cast<int>(h2vFlatVertexIds.size()));

    CBSTOperations v2hOps("V2H", params.payloadCapacity, params.alignment);
    v2hOps.construct(cbstV2HKeys, cbstV2HStartOffsets, numVertices,
                     v2hFlatHyperedgeIds.data(), static_cast<int>(v2hFlatHyperedgeIds.size()));

    CBSTOperations h2hOps("H2H", params.payloadCapacity, params.alignment);
    h2hOps.construct(cbstH2HKeys, cbstH2HStartOffsets, params.numHyperedges,
                     h2hFlatAdjacency.data(), static_cast<int>(h2hFlatAdjacency.size()));

    // Baseline motif counts
    computeMotifCounts(h2vOps.context(), v2hOps.context(), h2hOps.context(), params.numHyperedges);

    // --------------------------
    // DeltaGeneration()
    // --------------------------
    // Choose deletions: sample last few IDs for demo
    int N = params.numHyperedges;
    int numDeletes = std::max(1, std::min(2, N));
    std::vector<int> deletedIds;
    for (int k = 0; k < numDeletes; ++k) deletedIds.push_back(N - k);
    std::sort(deletedIds.begin(), deletedIds.end());

    // Generate insertions: a bit more than deletions
    int numInserts = numDeletes + 1;
    auto generatedInserts = hyperedge2vertex(numInserts, params.maxVerticesPerHyperedge, params.minVertexId, params.maxVertexId);
    // Assigned IDs: reuse smallest deleted first, then append new IDs after N
    int reuseK = std::min(numDeletes, numInserts);
    std::vector<int> insertAssignedIds(numInserts);
    for (int i = 0; i < reuseK; ++i) insertAssignedIds[i] = deletedIds[i];
    for (int i = reuseK; i < numInserts; ++i) insertAssignedIds[i] = N + (i - reuseK) + 1;

    // --------------------------
    // DataStructureUpdate() (host model for updated structures)
    // --------------------------
    // Build V2H removals: map vertex -> list of deleted hyperedge IDs
    std::unordered_map<int, std::vector<int>> vRem;
    for (int hId : deletedIds) {
        if (hId >= 1 && hId <= static_cast<int>(hyperedgeToVertex.size())) {
            for (int v : hyperedgeToVertex[hId - 1]) vRem[v].push_back(hId);
        }
    }
    std::vector<int> v2hRemoveKeys, v2hRemoveValues, v2hRemovePrefix;
    v2hRemoveKeys.reserve(vRem.size());
    for (auto &kv : vRem) {
        v2hRemoveKeys.push_back(kv.first);
        for (int h : kv.second) v2hRemoveValues.push_back(h);
        int newSize = (v2hRemovePrefix.empty() ? 0 : v2hRemovePrefix.back()) + static_cast<int>(kv.second.size());
        v2hRemovePrefix.push_back(newSize);
    }

    // Build V2H insertions: map vertex -> list of inserted (assigned) hyperedge IDs
    std::unordered_map<int, std::vector<int>> vIns;
    for (size_t i = 0; i < generatedInserts.size(); ++i) {
        int hId = insertAssignedIds[i];
        for (int v : generatedInserts[i]) vIns[v].push_back(hId);
    }
    std::vector<int> v2hInsertKeys, v2hInsertValues, v2hInsertPrefix;
    v2hInsertKeys.reserve(vIns.size());
    for (auto &kv : vIns) {
        v2hInsertKeys.push_back(kv.first);
        for (int h : kv.second) v2hInsertValues.push_back(h);
        int newSize = (v2hInsertPrefix.empty() ? 0 : v2hInsertPrefix.back()) + static_cast<int>(kv.second.size());
        v2hInsertPrefix.push_back(newSize);
    }

    // H2V delete on device
    h2vOps.erase(deletedIds);
    // V2H unfill on device
    if (!v2hRemoveKeys.empty()) {
        unfillCBST(v2hRemoveKeys, v2hRemoveValues, v2hRemovePrefix, const_cast<CBSTContext&>(v2hOps.context()));
    }

    // Prepare H2V insert payload vectors (keys, payload, prefix)
    std::vector<int> h2vInsertKeys = insertAssignedIds;
    std::vector<int> h2vInsertPayload;
    std::vector<int> h2vInsertPrefix;
    for (size_t i = 0; i < generatedInserts.size(); ++i) {
        for (int v : generatedInserts[i]) h2vInsertPayload.push_back(v);
        int newSize = (h2vInsertPrefix.empty() ? 0 : h2vInsertPrefix.back()) + static_cast<int>(generatedInserts[i].size());
        h2vInsertPrefix.push_back(newSize);
    }
    // H2V insert on device (reuses deleted IDs first, appends surplus)
    h2vOps.insert(h2vInsertKeys, h2vInsertPayload, h2vInsertPrefix);

    // V2H fill on device
    if (!v2hInsertKeys.empty()) {
        fillCBST(v2hInsertKeys, v2hInsertValues, v2hInsertPrefix, const_cast<CBSTContext&>(v2hOps.context()));
    }

    // Host updated structures for rebuild H2H
    // Apply deletions and insertions to host-side H2V representation
    int maxId = std::max(N, insertAssignedIds.empty() ? N : *std::max_element(insertAssignedIds.begin(), insertAssignedIds.end()));
    std::vector<std::vector<int>> updatedH2V = hyperedgeToVertex;
    if (static_cast<int>(updatedH2V.size()) < maxId) updatedH2V.resize(maxId);
    for (int hId : deletedIds) {
        if (hId >= 1 && hId <= static_cast<int>(updatedH2V.size())) updatedH2V[hId - 1].clear();
    }
    for (size_t i = 0; i < generatedInserts.size(); ++i) {
        int hId = insertAssignedIds[i];
        if (hId >= 1) {
            if (hId > static_cast<int>(updatedH2V.size())) updatedH2V.resize(hId);
            updatedH2V[hId - 1] = generatedInserts[i];
        }
    }
    // Rebuild V2H and H2H on host
    auto updatedV2H = vertex2hyperedge(updatedH2V);
    auto updatedH2H = hyperedgeAdjacency(updatedV2H, updatedH2V);

    // Flatten and reconstruct updated CBSTs
    auto [h2vFlatValsNew, h2vStartsNew] = flatten(updatedH2V, "Updated Hyperedge to Vertex");
    auto [v2hFlatValsNew, v2hStartsNew] = flatten(updatedV2H, "Updated Vertex to Hyperedge");
    auto [h2hFlatValsNew, h2hStartsNew] = flatten(updatedH2H, "Updated Hyperedge to Hyperedge");
    auto [cbstH2VStartsNew, cbstH2VKeysNew] = prepareCBSTData(h2vStartsNew);
    auto [cbstV2HStartsNew, cbstV2HKeysNew] = prepareCBSTData(v2hStartsNew);
    auto [cbstH2HStartsNew, cbstH2HKeysNew] = prepareCBSTData(h2hStartsNew);

    CBSTOperations h2vOpsNew("H2V-new", params.payloadCapacity, params.alignment);
    h2vOpsNew.construct(cbstH2VKeysNew, cbstH2VStartsNew, maxId, h2vFlatValsNew.data(), static_cast<int>(h2vFlatValsNew.size()));
    CBSTOperations v2hOpsNew("V2H-new", params.payloadCapacity, params.alignment);
    v2hOpsNew.construct(cbstV2HKeysNew, cbstV2HStartsNew, static_cast<int>(updatedV2H.size()), v2hFlatValsNew.data(), static_cast<int>(v2hFlatValsNew.size()));
    CBSTOperations h2hOpsNew("H2H-new", params.payloadCapacity, params.alignment);
    h2hOpsNew.construct(cbstH2HKeysNew, cbstH2HStartsNew, maxId, h2hFlatValsNew.data(), static_cast<int>(h2hFlatValsNew.size()));

    // --------------------------
    // CountUpdate(): subtract on deleted frontier (old), add on inserted frontier (new)
    // --------------------------
    std::vector<int> deltaCounts;
    computeMotifCountsDelta(h2vOps.context(), h2hOps.context(), h2vOpsNew.context(), h2hOpsNew.context(), deletedIds, insertAssignedIds, deltaCounts);
    std::cout << "Motif delta counts (30 bins): ";
    for (int i = 0; i < 30; ++i) std::cout << deltaCounts[i] << (i + 1 < 30 ? ' ' : '\n');

    // Clean up memory
    delete[] cbstH2VKeys;
    delete[] cbstV2HKeys;
    delete[] cbstH2HKeys;
    
    // --------------------------
    // Temporal motif counting (strict-inc) — gated by --temporal
    // --------------------------
    if (params.enableTemporal) {
        TemporalHypergraphIndex oldWin(params.payloadCapacity, params.alignment);
        TemporalHypergraphIndex newWin(params.payloadCapacity, params.alignment);
        // Synthetic temporal window (deterministic) to validate counts if requested
        if (params.temporalSynthetic) {
            // Build 3 hyperedges with shared vertices across layers to force counts
            // Older: 1:{1,2}, 2:{2,3}, 3:{1,3}
            // Middle: 1:{1,2}, 2:{2,3}, 3:{1,3}
            // Newest: 1:{1,2}, 2:{2,3}, 3:{1,3}
            std::vector<std::vector<int>> L = {{1,2},{2,3},{1,3}};
            auto [Lflat, Lstarts] = flatten(L, "Temporal Synthetic H2V Layer");
            auto [cbstStarts, cbstKeys] = prepareCBSTData(Lstarts);
            // Older/Middle/Newest for oldWin are same synthetic for baseline
            oldWin.h2v.construct(TemporalLayer::Older,  cbstKeys, cbstStarts, 3, Lflat.data(), static_cast<int>(Lflat.size()));
            oldWin.h2v.construct(TemporalLayer::Middle, cbstKeys, cbstStarts, 3, Lflat.data(), static_cast<int>(Lflat.size()));
            oldWin.h2v.construct(TemporalLayer::Newest, cbstKeys, cbstStarts, 3, Lflat.data(), static_cast<int>(Lflat.size()));
            // Build V2H for synthetic layers
            auto Lv2h = vertex2hyperedge(L);
            auto [Lv2hFlat, Lv2hStarts] = flatten(Lv2h, "Temporal Synthetic V2H Layer");
            auto [cbstLv2hStarts, cbstLv2hKeys] = prepareCBSTData(Lv2hStarts);
            int numV_L = static_cast<int>(Lv2h.size());
            oldWin.v2h.construct(TemporalLayer::Older,  cbstLv2hKeys, cbstLv2hStarts, numV_L, Lv2hFlat.data(), static_cast<int>(Lv2hFlat.size()));
            oldWin.v2h.construct(TemporalLayer::Middle, cbstLv2hKeys, cbstLv2hStarts, numV_L, Lv2hFlat.data(), static_cast<int>(Lv2hFlat.size()));
            oldWin.v2h.construct(TemporalLayer::Newest, cbstLv2hKeys, cbstLv2hStarts, numV_L, Lv2hFlat.data(), static_cast<int>(Lv2hFlat.size()));
            // New window: change Newest to break/add edges to induce non-zero delta
            std::vector<std::vector<int>> L2 = {{1,2},{2,3},{1,2,3}}; // add vertex 2 to edge 3
            auto [L2flat, L2starts] = flatten(L2, "Temporal Synthetic H2V Layer Newest");
            auto [cbstStarts2, cbstKeys2] = prepareCBSTData(L2starts);
            newWin.h2v.construct(TemporalLayer::Older,  cbstKeys,  cbstStarts, 3, Lflat.data(),  static_cast<int>(Lflat.size()));
            newWin.h2v.construct(TemporalLayer::Middle, cbstKeys,  cbstStarts, 3, Lflat.data(),  static_cast<int>(Lflat.size()));
            newWin.h2v.construct(TemporalLayer::Newest, cbstKeys2, cbstStarts2,3, L2flat.data(), static_cast<int>(L2flat.size()));
            // V2H for new window: Older/Middle from L, Newest from L2
            auto L2v2h = vertex2hyperedge(L2);
            auto [L2v2hFlat, L2v2hStarts] = flatten(L2v2h, "Temporal Synthetic V2H Layer Newest");
            auto [cbstL2v2hStarts, cbstL2v2hKeys] = prepareCBSTData(L2v2hStarts);
            int numV_L2 = static_cast<int>(L2v2h.size());
            newWin.v2h.construct(TemporalLayer::Older,  cbstLv2hKeys,  cbstLv2hStarts, numV_L,  Lv2hFlat.data(),  static_cast<int>(Lv2hFlat.size()));
            newWin.v2h.construct(TemporalLayer::Middle, cbstLv2hKeys,  cbstLv2hStarts, numV_L,  Lv2hFlat.data(),  static_cast<int>(Lv2hFlat.size()));
            newWin.v2h.construct(TemporalLayer::Newest, cbstL2v2hKeys, cbstL2v2hStarts, numV_L2, L2v2hFlat.data(), static_cast<int>(L2v2hFlat.size()));
            // Optionally apply extra deltas on top of the synthetic Newest
            if (!params.temporalDeltasPath.empty()) {
                applyTemporalDeltasFromFile(params.temporalDeltasPath, newWin);
            }
        } else {
            auto [tH2VStartsVals, tH2VKeys] = prepareCBSTData(h2vStartOffsets);
            oldWin.h2v.construct(TemporalLayer::Older,
                          tH2VKeys, tH2VStartsVals, params.numHyperedges,
                          h2vFlatVertexIds.data(), static_cast<int>(h2vFlatVertexIds.size()));
            oldWin.h2v.construct(TemporalLayer::Middle,
                          tH2VKeys, tH2VStartsVals, params.numHyperedges,
                          h2vFlatVertexIds.data(), static_cast<int>(h2vFlatVertexIds.size()));
            oldWin.h2v.construct(TemporalLayer::Newest,
                          tH2VKeys, tH2VStartsVals, params.numHyperedges,
                          h2vFlatVertexIds.data(), static_cast<int>(h2vFlatVertexIds.size()));
            // Also construct V2H for baseline layers
            auto [tV2HStartsVals, tV2HKeys] = prepareCBSTData(v2hStartOffsets);
            oldWin.v2h.construct(TemporalLayer::Older,
                          tV2HKeys, tV2HStartsVals, numVertices,
                          v2hFlatHyperedgeIds.data(), static_cast<int>(v2hFlatHyperedgeIds.size()));
            oldWin.v2h.construct(TemporalLayer::Middle,
                          tV2HKeys, tV2HStartsVals, numVertices,
                          v2hFlatHyperedgeIds.data(), static_cast<int>(v2hFlatHyperedgeIds.size()));
            oldWin.v2h.construct(TemporalLayer::Newest,
                          tV2HKeys, tV2HStartsVals, numVertices,
                          v2hFlatHyperedgeIds.data(), static_cast<int>(v2hFlatHyperedgeIds.size()));
        }

        // Baseline window count
        std::vector<int> temporalCounts;
        computeTemporalMotifCountsStrictInc(oldWin, temporalCounts);
        std::cout << "Temporal strict-inc counts (30 bins): ";
        for (int i = 0; i < 30; ++i) std::cout << temporalCounts[i] << (i + 1 < 30 ? ' ' : '\n');

        if (!params.temporalSynthetic) {
            // Phase 3: simulate window rotation where Newest changes to updated state
            // Old window (A,B,C) all baseline; New window (B,C,D) with D = updated
            auto [tH2VStartsVals, tH2VKeys] = prepareCBSTData(h2vStartOffsets);
            newWin.h2v.construct(TemporalLayer::Older,
                                 tH2VKeys, tH2VStartsVals, params.numHyperedges,
                                 h2vFlatVertexIds.data(), static_cast<int>(h2vFlatVertexIds.size()));
            newWin.h2v.construct(TemporalLayer::Middle,
                                 tH2VKeys, tH2VStartsVals, params.numHyperedges,
                                 h2vFlatVertexIds.data(), static_cast<int>(h2vFlatVertexIds.size()));
            newWin.h2v.construct(TemporalLayer::Newest,
                                 cbstH2VKeysNew, cbstH2VStartsNew, maxId,
                                 h2vFlatValsNew.data(), static_cast<int>(h2vFlatValsNew.size()));
            // Construct V2H for new window: Older/Middle baseline, Newest updated
            auto [tV2HStartsVals, tV2HKeys] = prepareCBSTData(v2hStartOffsets);
            newWin.v2h.construct(TemporalLayer::Older,
                                 tV2HKeys, tV2HStartsVals, numVertices,
                                 v2hFlatHyperedgeIds.data(), static_cast<int>(v2hFlatHyperedgeIds.size()));
            newWin.v2h.construct(TemporalLayer::Middle,
                                 tV2HKeys, tV2HStartsVals, numVertices,
                                 v2hFlatHyperedgeIds.data(), static_cast<int>(v2hFlatHyperedgeIds.size()));
            newWin.v2h.construct(TemporalLayer::Newest,
                                 cbstV2HKeysNew, cbstV2HStartsNew, static_cast<int>(updatedV2H.size()),
                                 v2hFlatValsNew.data(), static_cast<int>(v2hFlatValsNew.size()));

            // Optional: apply external deltas onto Newest if provided
            if (!params.temporalDeltasPath.empty()) {
                // Rotate oldWin to simulate (B,C,D) from (A,B,C)
                oldWin.rotate();
                oldWin.h2v.construct(TemporalLayer::Newest,
                                     cbstH2VKeysNew, cbstH2VStartsNew, maxId,
                                     h2vFlatValsNew.data(), static_cast<int>(h2vFlatValsNew.size()));
                oldWin.v2h.construct(TemporalLayer::Newest,
                                     cbstV2HKeysNew, cbstV2HStartsNew, static_cast<int>(updatedV2H.size()),
                                     v2hFlatValsNew.data(), static_cast<int>(v2hFlatValsNew.size()));
                // Apply deltas onto newWin Newest
                applyTemporalDeltasFromFile(params.temporalDeltasPath, newWin);
            }
        }

        std::vector<int> deltaCounts;
        computeTemporalMotifCountsStrictIncDelta(oldWin, newWin, deltaCounts);
        std::cout << "Temporal strict-inc delta (30 bins): ";
        for (int i = 0; i < 30; ++i) std::cout << deltaCounts[i] << (i + 1 < 30 ? ' ' : '\n');
    }

    return 0;
}