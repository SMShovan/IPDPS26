#include <algorithm>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../include/graphGeneration.hpp"
#include "../../../include/motif_update.hpp"
#include "../../../include/structure.hpp"
#include "../../../include/utils.hpp"

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

const std::vector<std::pair<std::string, std::string>> kDatasets = {
    {"Coauth", "Coauth.txt"},
    {"Orkut", "Orkut.txt"},
    {"Random", "Random.txt"},
    {"Tags", "Tags.txt"},
    {"Threads", "Threads.txt"},
};
const std::vector<int> kCardinalities = {50, 100, 200};
constexpr int kAlignment = 4;
constexpr int kPayloadCapacity = 1 << 20;
constexpr int kTotalChanges = 100000;
constexpr int kMinVertexId = 1;
constexpr int kDefaultMaxVertexId = 1000000;

int nextMultipleOf4(int value) {
    return ((value + 3) / 4) * 4;
}

bool loadHyperedgeListFromInputFile(const std::string& inputFileName, std::vector<std::vector<int>>& h2v) {
    const std::string path = std::string("../../../input/") + inputFileName;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error: failed to open dataset file: " << path << std::endl;
        return false;
    }
    h2v.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::vector<int> row;
        int v = 0;
        while (iss >> v) {
            if (v <= 0) {
                std::cerr << "Error: vertex IDs must be positive in " << path << std::endl;
                return false;
            }
            row.push_back(v);
        }
        if (!row.empty()) h2v.push_back(std::move(row));
    }
    return !h2v.empty();
}

PreparedCBSTData prepareSingleCBSTInput(const std::vector<int>& startOffsets) {
    auto [preparedStartOffsets, preparedKeys] = prepareCBSTData(startOffsets);
    PreparedCBSTData data;
    data.startOffsets = preparedStartOffsets;
    data.keys.reset(preparedKeys);
    return data;
}

HostGraphState buildHostGraphState(const std::vector<std::vector<int>>& h2v, const std::vector<std::vector<int>>& v2h, const std::string& labelPrefix) {
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

DeviceGraphState constructDeviceGraphState(const HostGraphState& hostState, int payloadCapacity, int alignment, int h2vNumRecords, int h2hNumRecords,
                                           const char* h2vName, const char* v2hName, const char* h2hName) {
    auto h2vPrepared = prepareSingleCBSTInput(hostState.h2vStartOffsets);
    auto v2hPrepared = prepareSingleCBSTInput(hostState.v2hStartOffsets);
    auto h2hPrepared = prepareSingleCBSTInput(hostState.h2hStartOffsets);

    DeviceGraphState deviceState(h2vName, v2hName, h2hName, payloadCapacity, alignment);
    deviceState.h2vOps.construct(h2vPrepared.keys.get(), h2vPrepared.startOffsets, h2vNumRecords,
                                 const_cast<int*>(hostState.h2vFlatValues.data()), static_cast<int>(hostState.h2vFlatValues.size()));
    deviceState.v2hOps.construct(v2hPrepared.keys.get(), v2hPrepared.startOffsets, static_cast<int>(hostState.v2h.size()),
                                 const_cast<int*>(hostState.v2hFlatValues.data()), static_cast<int>(hostState.v2hFlatValues.size()));
    deviceState.h2hOps.construct(h2hPrepared.keys.get(), h2hPrepared.startOffsets, h2hNumRecords,
                                 const_cast<int*>(hostState.h2hFlatValues.data()), static_cast<int>(hostState.h2hFlatValues.size()));
    return deviceState;
}

std::vector<std::vector<int>> generateFixedCardinalityHyperedges(int count, int cardinality, int minVertexId, int maxVertexId) {
    std::vector<std::vector<int>> inserts;
    inserts.reserve(count);
    std::mt19937 rng(12345U + static_cast<unsigned>(count) + static_cast<unsigned>(cardinality));
    std::uniform_int_distribution<int> dist(minVertexId, maxVertexId);

    for (int i = 0; i < count; ++i) {
        std::unordered_set<int> unique;
        while (static_cast<int>(unique.size()) < cardinality) {
            unique.insert(dist(rng));
        }
        std::vector<int> row(unique.begin(), unique.end());
        std::sort(row.begin(), row.end());
        inserts.push_back(std::move(row));
    }
    return inserts;
}

UpdateBatch buildSyntheticUpdateBatch(const HypergraphParams& params, const std::vector<std::vector<int>>& originalH2V, int changedEdgeCardinality) {
    UpdateBatch batch;
    int numHyperedges = params.numHyperedges;
    int numDeletes = static_cast<int>(std::llround(static_cast<double>(params.totalChanges) * (params.deletionPercentage / 100.0)));
    int numInserts = static_cast<int>(std::llround(static_cast<double>(params.totalChanges) * (params.insertionPercentage / 100.0)));

    int totalFromPercentages = numDeletes + numInserts;
    if (totalFromPercentages != params.totalChanges) {
        int diff = params.totalChanges - totalFromPercentages;
        if (params.insertionPercentage >= params.deletionPercentage) numInserts += diff;
        else numDeletes += diff;
    }
    if (numDeletes > numHyperedges) {
        numDeletes = numHyperedges;
        numInserts = std::max(0, params.totalChanges - numDeletes);
    }
    if (numDeletes < 0) numDeletes = 0;
    if (numInserts < 0) numInserts = 0;

    for (int k = 0; k < numDeletes; ++k) batch.deletedIds.push_back(numHyperedges - k);
    std::sort(batch.deletedIds.begin(), batch.deletedIds.end());

    int maxVertexId = std::max(params.maxVertexId, changedEdgeCardinality * 20);
    batch.generatedInserts = generateFixedCardinalityHyperedges(numInserts, changedEdgeCardinality, params.minVertexId, maxVertexId);

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
    int maxAssignedId =
        updateBatch.insertAssignedIds.empty() ? originalSize : *std::max_element(updateBatch.insertAssignedIds.begin(), updateBatch.insertAssignedIds.end());
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

bool writeOutput(const std::vector<std::tuple<std::string, int, double>>& rows) {
    std::ofstream out("outputHyperedgeCardinality.txt");
    if (!out.is_open()) {
        std::cerr << "Error: cannot write outputHyperedgeCardinality.txt" << std::endl;
        return false;
    }
    out << "Dataset Cardinality TimeSeconds\n";
    for (const auto& row : rows) {
        out << std::get<0>(row) << " " << std::get<1>(row) << " " << std::get<2>(row) << "\n";
    }
    return true;
}

} // namespace

std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::vector<int> flatValues;
    std::vector<int> startOffsets(vec2d.size());
    int index = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        startOffsets[i] = index;
        int innerSize = static_cast<int>(vec2d[i].size());
        int paddedSize = nextMultipleOf4(innerSize);
        for (int j = 0; j < paddedSize; ++j) {
            if (j < innerSize) flatValues.push_back(vec2d[i][j]);
            else if (j == paddedSize - 1) flatValues.push_back(INT_MIN);
            else flatValues.push_back(0);
            ++index;
        }
    }
    return {flatValues, startOffsets};
}

int main() {
    std::vector<std::tuple<std::string, int, double>> outputRows;

    for (const auto& dataset : kDatasets) {
        const std::string datasetName = dataset.first;
        const std::string fileName = dataset.second;

        std::vector<std::vector<int>> h2v;
        if (!loadHyperedgeListFromInputFile(fileName, h2v)) return 1;
        std::vector<std::vector<int>> v2h = vertex2hyperedge(h2v);

        int datasetMinVertex = INT_MAX;
        int datasetMaxVertex = 0;
        for (const auto& row : h2v) {
            for (int v : row) {
                datasetMinVertex = std::min(datasetMinVertex, v);
                datasetMaxVertex = std::max(datasetMaxVertex, v);
            }
        }
        if (datasetMinVertex == INT_MAX) datasetMinVertex = kMinVertexId;
        if (datasetMaxVertex <= datasetMinVertex) datasetMaxVertex = std::max(datasetMinVertex + 1, kDefaultMaxVertexId);

        for (int cardinality : kCardinalities) {
            HypergraphParams params{};
            params.numHyperedges = static_cast<int>(h2v.size());
            params.maxVerticesPerHyperedge = cardinality;
            params.minVertexId = datasetMinVertex;
            params.maxVertexId = std::max(datasetMaxVertex, cardinality * 20);
            params.payloadCapacity = kPayloadCapacity;
            params.alignment = kAlignment;
            params.totalChanges = kTotalChanges;
            params.insertionPercentage = 50.0;
            params.deletionPercentage = 50.0;

            HostGraphState initialHostState = buildHostGraphState(h2v, v2h, "");
            DeviceGraphState oldDeviceState =
                constructDeviceGraphState(initialHostState, params.payloadCapacity, params.alignment, params.numHyperedges, params.numHyperedges,
                                          "H2V-old", "V2H-old", "H2H-old");
            DeviceGraphState mutableDeviceState =
                constructDeviceGraphState(initialHostState, params.payloadCapacity, params.alignment, params.numHyperedges, params.numHyperedges,
                                          "H2V-mut", "V2H-mut", "H2H-mut");

            auto t0 = std::chrono::high_resolution_clock::now();
            UpdateBatch updateBatch = buildSyntheticUpdateBatch(params, initialHostState.h2v, cardinality);
            auto t1 = std::chrono::high_resolution_clock::now();
            applyIncrementalUpdates(mutableDeviceState, updateBatch);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<int>> updatedH2V = buildUpdatedH2V(initialHostState.h2v, updateBatch);
            std::vector<std::vector<int>> updatedV2H = vertex2hyperedge(updatedH2V);
            HostGraphState updatedHostState = buildHostGraphState(updatedH2V, updatedV2H, "Updated ");
            int maxId = static_cast<int>(updatedHostState.h2v.size());
            DeviceGraphState updatedDeviceState =
                constructDeviceGraphState(updatedHostState, params.payloadCapacity, params.alignment, maxId, maxId, "H2V-new", "V2H-new", "H2H-new");

            std::vector<int> deltaCounts;
            computeMotifCountsDelta(oldDeviceState.h2vOps.context(), oldDeviceState.h2hOps.context(), updatedDeviceState.h2vOps.context(),
                                    updatedDeviceState.h2hOps.context(), updateBatch.deletedIds, updateBatch.insertAssignedIds, deltaCounts);
            auto t3 = std::chrono::high_resolution_clock::now();

            double totalTimed = std::chrono::duration<double>(t1 - t0).count() + std::chrono::duration<double>(t2 - t1).count() +
                                std::chrono::duration<double>(t3 - t2).count();
            outputRows.push_back({datasetName, cardinality, totalTimed});
            std::cout << datasetName << " cardinality=" << cardinality << " totalTimed=" << totalTimed << " s\n";
        }
    }

    if (!writeOutput(outputRows)) return 1;
    std::cout << "figure6c benchmark complete." << std::endl;
    return 0;
}
