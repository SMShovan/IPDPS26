#include <algorithm>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../include/graphGeneration.hpp"
#include "../../../include/temporal_adjacency.hpp"
#include "../../../include/temporal_count.hpp"
#include "../../../include/utils.hpp"

namespace {

struct PreparedCBSTData {
    int* startOffsets = nullptr;
    std::unique_ptr<int[]> keys;
};

struct TemporalHostState {
    std::vector<std::vector<int>> h2v;
    std::vector<std::vector<int>> v2h;
    std::vector<int> h2vFlatValues;
    std::vector<int> h2vStartOffsets;
    std::vector<int> v2hFlatValues;
    std::vector<int> v2hStartOffsets;
};

struct UpdateBatch {
    std::vector<int> deletedIds;
    std::vector<int> insertAssignedIds;
    std::vector<std::vector<int>> generatedInserts;
};

struct DatasetResult {
    std::string datasetName;
    std::vector<int> batchSizes;
    std::vector<double> totalTimedSeconds;
};

constexpr int kDefaultAlignment = 4;
constexpr int kDefaultPayloadCapacity = 1 << 20;
const std::vector<int> kBatchSizes = {50000, 100000, 200000};
const std::vector<std::pair<std::string, std::string>> kDatasets = {
    {"Coauth", "Coauth.txt"},
    {"Tags", "Tags.txt"},
    {"Orkut", "Orkut.txt"},
    {"Threads", "Threads.txt"},
    {"Random", "Random.txt"},
};

bool loadHyperedgeListFromSharedInput(const std::string& inputFileName, std::vector<std::vector<int>>& h2v) {
    // Shared dataset folder from dynamicHyperGraph, requested by user.
    const std::string path = std::string("../../../../dynamicHyperGraph/input/") + inputFileName;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error: failed to open dataset file: " << path << std::endl;
        return false;
    }

    h2v.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::vector<int> row;
        int vertex = 0;
        while (iss >> vertex) {
            if (vertex <= 0) {
                std::cerr << "Error: vertex IDs must be positive in " << path << std::endl;
                return false;
            }
            row.push_back(vertex);
        }
        if (!row.empty()) {
            h2v.push_back(std::move(row));
        }
    }

    if (h2v.empty()) {
        std::cerr << "Error: empty hyperedge list in " << path << std::endl;
        return false;
    }
    return true;
}

PreparedCBSTData prepareSingleCBSTInput(const std::vector<int>& startOffsets) {
    auto [preparedStartOffsets, preparedKeys] = prepareCBSTData(startOffsets);
    PreparedCBSTData data;
    data.startOffsets = preparedStartOffsets;
    data.keys.reset(preparedKeys);
    return data;
}

TemporalHostState buildTemporalHostState(const std::vector<std::vector<int>>& h2v, const std::string& prefix) {
    TemporalHostState state;
    state.h2v = h2v;
    state.v2h = vertex2hyperedge(state.h2v);
    auto [h2vFlat, h2vStarts] = flatten(state.h2v, prefix + "Hyperedge to Vertex");
    auto [v2hFlat, v2hStarts] = flatten(state.v2h, prefix + "Vertex to Hyperedge");
    state.h2vFlatValues = std::move(h2vFlat);
    state.h2vStartOffsets = std::move(h2vStarts);
    state.v2hFlatValues = std::move(v2hFlat);
    state.v2hStartOffsets = std::move(v2hStarts);
    return state;
}

void constructLayer(TemporalHypergraphIndex& index,
                    TemporalLayer layer,
                    const TemporalHostState& state,
                    int h2vNumRecords,
                    int v2hNumRecords) {
    auto h2vPrepared = prepareSingleCBSTInput(state.h2vStartOffsets);
    auto v2hPrepared = prepareSingleCBSTInput(state.v2hStartOffsets);
    index.h2v.construct(layer, h2vPrepared.keys.get(), h2vPrepared.startOffsets, h2vNumRecords,
                        const_cast<int*>(state.h2vFlatValues.data()), static_cast<int>(state.h2vFlatValues.size()));
    index.v2h.construct(layer, v2hPrepared.keys.get(), v2hPrepared.startOffsets, v2hNumRecords,
                        const_cast<int*>(state.v2hFlatValues.data()), static_cast<int>(state.v2hFlatValues.size()));
}

UpdateBatch buildSyntheticUpdateBatch(int totalChanges,
                                      const std::vector<std::vector<int>>& originalH2V,
                                      int maxVerticesPerHyperedge,
                                      int minVertexId,
                                      int maxVertexId) {
    UpdateBatch batch;
    const int numHyperedges = static_cast<int>(originalH2V.size());
    if (numHyperedges <= 0 || totalChanges <= 0) {
        return batch;
    }

    const int effectiveChanges = std::min(totalChanges, numHyperedges);
    int numDeletes = effectiveChanges / 2;
    int numInserts = effectiveChanges - numDeletes;
    if (numDeletes == 0 && numInserts == 0) {
        numInserts = 1;
    }

    for (int k = 0; k < numDeletes; ++k) {
        batch.deletedIds.push_back(numHyperedges - k);
    }
    std::sort(batch.deletedIds.begin(), batch.deletedIds.end());

    batch.generatedInserts = hyperedge2vertex(numInserts, std::max(1, maxVerticesPerHyperedge),
                                              std::max(1, minVertexId), std::max(minVertexId + 1, maxVertexId));

    const int reuseCount = std::min(numDeletes, numInserts);
    batch.insertAssignedIds.resize(numInserts);
    for (int i = 0; i < reuseCount; ++i) {
        batch.insertAssignedIds[i] = batch.deletedIds[i];
    }
    for (int i = reuseCount; i < numInserts; ++i) {
        batch.insertAssignedIds[i] = numHyperedges + (i - reuseCount) + 1;
    }
    return batch;
}

std::vector<std::vector<int>> buildUpdatedH2V(const std::vector<std::vector<int>>& originalH2V, const UpdateBatch& batch) {
    const int originalSize = static_cast<int>(originalH2V.size());
    int maxAssignedId =
        batch.insertAssignedIds.empty() ? originalSize : *std::max_element(batch.insertAssignedIds.begin(), batch.insertAssignedIds.end());
    int maxId = std::max(originalSize, maxAssignedId);

    std::vector<std::vector<int>> updatedH2V = originalH2V;
    if (static_cast<int>(updatedH2V.size()) < maxId) {
        updatedH2V.resize(maxId);
    }
    for (int hId : batch.deletedIds) {
        if (hId >= 1 && hId <= static_cast<int>(updatedH2V.size())) {
            updatedH2V[hId - 1].clear();
        }
    }
    for (size_t i = 0; i < batch.generatedInserts.size(); ++i) {
        int hId = batch.insertAssignedIds[i];
        if (hId >= 1) {
            if (hId > static_cast<int>(updatedH2V.size())) {
                updatedH2V.resize(hId);
            }
            updatedH2V[hId - 1] = batch.generatedInserts[i];
        }
    }
    return updatedH2V;
}

bool writeOutputFile(const DatasetResult& result) {
    const std::string path = "output" + result.datasetName + ".txt";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Error: could not write " << path << std::endl;
        return false;
    }
    out << "ChangedEdges TimeSeconds\n";
    for (size_t i = 0; i < result.batchSizes.size(); ++i) {
        out << result.batchSizes[i] << " " << result.totalTimedSeconds[i] << "\n";
    }
    return true;
}

}  // namespace

int main() {
    for (const auto& dataset : kDatasets) {
        const std::string& datasetName = dataset.first;
        const std::string& inputFile = dataset.second;

        std::vector<std::vector<int>> h2v;
        if (!loadHyperedgeListFromSharedInput(inputFile, h2v)) {
            return 1;
        }

        int maxVerticesPerHyperedge = 1;
        int minVertexId = INT_MAX;
        int maxVertexId = 0;
        for (const auto& row : h2v) {
            maxVerticesPerHyperedge = std::max(maxVerticesPerHyperedge, static_cast<int>(row.size()));
            for (int v : row) {
                minVertexId = std::min(minVertexId, v);
                maxVertexId = std::max(maxVertexId, v);
            }
        }
        if (minVertexId == INT_MAX) {
            minVertexId = 1;
        }
        if (maxVertexId < minVertexId) {
            maxVertexId = minVertexId + 1;
        }

        DatasetResult result;
        result.datasetName = datasetName;
        result.batchSizes = kBatchSizes;

        std::cout << "Running figure12a benchmark on " << datasetName << std::endl;
        for (int batchSize : kBatchSizes) {
            auto t0 = std::chrono::high_resolution_clock::now();

            TemporalHostState baselineState = buildTemporalHostState(h2v, "");
            const int baselineNumEdges = static_cast<int>(baselineState.h2v.size());
            const int baselineNumVertices = static_cast<int>(baselineState.v2h.size());

            TemporalHypergraphIndex oldWindow(kDefaultPayloadCapacity, kDefaultAlignment);
            constructLayer(oldWindow, TemporalLayer::Older, baselineState, baselineNumEdges, baselineNumVertices);
            constructLayer(oldWindow, TemporalLayer::Middle, baselineState, baselineNumEdges, baselineNumVertices);
            constructLayer(oldWindow, TemporalLayer::Newest, baselineState, baselineNumEdges, baselineNumVertices);

            auto t1 = std::chrono::high_resolution_clock::now();

            UpdateBatch updateBatch = buildSyntheticUpdateBatch(batchSize, baselineState.h2v, maxVerticesPerHyperedge, minVertexId, maxVertexId);
            std::vector<std::vector<int>> updatedH2V = buildUpdatedH2V(baselineState.h2v, updateBatch);
            TemporalHostState updatedState = buildTemporalHostState(updatedH2V, "Updated ");

            const int updatedNumEdges = static_cast<int>(updatedState.h2v.size());
            const int updatedNumVertices = static_cast<int>(updatedState.v2h.size());

            TemporalHypergraphIndex newWindow(kDefaultPayloadCapacity, kDefaultAlignment);
            constructLayer(newWindow, TemporalLayer::Older, baselineState, baselineNumEdges, baselineNumVertices);
            constructLayer(newWindow, TemporalLayer::Middle, baselineState, baselineNumEdges, baselineNumVertices);
            constructLayer(newWindow, TemporalLayer::Newest, updatedState, updatedNumEdges, updatedNumVertices);

            auto t2 = std::chrono::high_resolution_clock::now();

            std::vector<int> deltaCounts;
            computeTemporalMotifCountsStrictIncDelta(oldWindow, newWindow, deltaCounts);

            auto t3 = std::chrono::high_resolution_clock::now();
            const double constructSeconds = std::chrono::duration<double>(t1 - t0).count();
            const double updateSeconds = std::chrono::duration<double>(t2 - t1).count();
            const double deltaSeconds = std::chrono::duration<double>(t3 - t2).count();
            const double totalTimed = constructSeconds + updateSeconds + deltaSeconds;

            result.totalTimedSeconds.push_back(totalTimed);
            std::cout << "  batch=" << batchSize
                      << " construct=" << constructSeconds
                      << " update=" << updateSeconds
                      << " delta=" << deltaSeconds
                      << " total=" << totalTimed << " s" << std::endl;
        }

        if (!writeOutputFile(result)) {
            return 1;
        }
    }

    std::cout << "figure12a benchmarking complete." << std::endl;
    return 0;
}
