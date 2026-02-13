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

struct IncidentUpdateBatch {
    std::vector<int> addHyperedgeIds;
    std::vector<int> addVertices;
    std::vector<int> addPrefix;
    std::vector<int> removeHyperedgeIds;
    std::vector<int> removeVertices;
    std::vector<int> removePrefix;

    std::vector<int> v2hInsertKeys;
    std::vector<int> v2hInsertValues;
    std::vector<int> v2hInsertPrefix;
    std::vector<int> v2hRemoveKeys;
    std::vector<int> v2hRemoveValues;
    std::vector<int> v2hRemovePrefix;
};

const std::vector<std::pair<std::string, std::string>> kDatasets = {
    {"Coauth", "Coauth.txt"},
    {"Tags", "Tags.txt"},
    {"Orkut", "Orkut.txt"},
    {"Threads", "Threads.txt"},
    {"Random", "Random.txt"},
};
const std::vector<int> kIncidentSizes = {50000, 100000, 200000};
constexpr int kAlignment = 4;
constexpr int kPayloadCapacity = 1 << 20;
constexpr int kMinVertexId = 1;
constexpr int kDefaultMaxVertexId = 1000000;
constexpr double kInsertionPercentage = 50.0;
constexpr double kDeletionPercentage = 50.0;

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

IncidentUpdateBatch buildIncidentUpdateBatch(const HypergraphParams& params, const std::vector<std::vector<int>>& originalH2V, int incidentChangeCount) {
    IncidentUpdateBatch batch;
    int numHyperedges = static_cast<int>(originalH2V.size());
    int adds = static_cast<int>(std::llround(static_cast<double>(incidentChangeCount) * (kInsertionPercentage / 100.0)));
    int removes = incidentChangeCount - adds;
    if (adds < 0) adds = 0;
    if (removes < 0) removes = 0;

    std::mt19937 rng(2026U + static_cast<unsigned>(incidentChangeCount));
    std::uniform_int_distribution<int> hedgeDist(1, std::max(1, numHyperedges));
    int maxVertexId = std::max(params.maxVertexId, kDefaultMaxVertexId);
    std::uniform_int_distribution<int> vtxDist(std::max(1, params.minVertexId), std::max(std::max(1, params.minVertexId), maxVertexId));

    std::vector<std::unordered_set<int>> currentSets(numHyperedges);
    for (int h = 0; h < numHyperedges; ++h) {
        currentSets[h] = std::unordered_set<int>(originalH2V[h].begin(), originalH2V[h].end());
    }

    batch.addHyperedgeIds.reserve(adds);
    batch.removeHyperedgeIds.reserve(removes);

    for (int i = 0; i < adds; ++i) {
        int hId = hedgeDist(rng);
        int v = vtxDist(rng);
        while (currentSets[hId - 1].count(v) > 0) v = vtxDist(rng);
        currentSets[hId - 1].insert(v);
        batch.addHyperedgeIds.push_back(hId);
        batch.addVertices.push_back(v);
        batch.addPrefix.push_back(static_cast<int>(batch.addVertices.size()));
    }

    for (int i = 0; i < removes; ++i) {
        int hId = hedgeDist(rng);
        if (currentSets[hId - 1].empty()) continue;
        int v = *currentSets[hId - 1].begin();
        currentSets[hId - 1].erase(v);
        batch.removeHyperedgeIds.push_back(hId);
        batch.removeVertices.push_back(v);
        batch.removePrefix.push_back(static_cast<int>(batch.removeVertices.size()));
    }

    std::unordered_map<int, std::vector<int>> v2hAdd;
    for (size_t i = 0; i < batch.addVertices.size(); ++i) {
        v2hAdd[batch.addVertices[i]].push_back(batch.addHyperedgeIds[i]);
    }
    for (auto& kv : v2hAdd) {
        batch.v2hInsertKeys.push_back(kv.first);
        for (int hId : kv.second) batch.v2hInsertValues.push_back(hId);
        int newSize = (batch.v2hInsertPrefix.empty() ? 0 : batch.v2hInsertPrefix.back()) + static_cast<int>(kv.second.size());
        batch.v2hInsertPrefix.push_back(newSize);
    }

    std::unordered_map<int, std::vector<int>> v2hRemove;
    for (size_t i = 0; i < batch.removeVertices.size(); ++i) {
        v2hRemove[batch.removeVertices[i]].push_back(batch.removeHyperedgeIds[i]);
    }
    for (auto& kv : v2hRemove) {
        batch.v2hRemoveKeys.push_back(kv.first);
        for (int hId : kv.second) batch.v2hRemoveValues.push_back(hId);
        int newSize = (batch.v2hRemovePrefix.empty() ? 0 : batch.v2hRemovePrefix.back()) + static_cast<int>(kv.second.size());
        batch.v2hRemovePrefix.push_back(newSize);
    }

    return batch;
}

void applyIncidentVertexUpdates(DeviceGraphState& deviceState, const IncidentUpdateBatch& batch) {
    if (!batch.v2hRemoveKeys.empty()) {
        unfillCBST(batch.v2hRemoveKeys, batch.v2hRemoveValues, batch.v2hRemovePrefix, const_cast<CBSTContext&>(deviceState.v2hOps.context()));
    }
    if (!batch.v2hInsertKeys.empty()) {
        fillCBST(batch.v2hInsertKeys, batch.v2hInsertValues, batch.v2hInsertPrefix, const_cast<CBSTContext&>(deviceState.v2hOps.context()));
    }
}

std::vector<std::vector<int>> applyIncidentChangesOnHost(const std::vector<std::vector<int>>& originalH2V, const IncidentUpdateBatch& batch) {
    std::vector<std::vector<int>> updated = originalH2V;
    for (size_t i = 0; i < batch.removeVertices.size(); ++i) {
        int hId = batch.removeHyperedgeIds[i];
        int v = batch.removeVertices[i];
        if (hId < 1 || hId > static_cast<int>(updated.size())) continue;
        auto& row = updated[hId - 1];
        row.erase(std::remove(row.begin(), row.end(), v), row.end());
    }
    for (size_t i = 0; i < batch.addVertices.size(); ++i) {
        int hId = batch.addHyperedgeIds[i];
        int v = batch.addVertices[i];
        if (hId < 1 || hId > static_cast<int>(updated.size())) continue;
        auto& row = updated[hId - 1];
        row.push_back(v);
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return updated;
}

bool writeOutput(const std::vector<std::tuple<std::string, std::string, double>>& rows) {
    std::ofstream out("outputIncidentVertex.txt");
    if (!out.is_open()) {
        std::cerr << "Error: cannot write outputIncidentVertex.txt" << std::endl;
        return false;
    }
    out << "Dataset Modification TimeMilliseconds\n";
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
    std::vector<std::tuple<std::string, std::string, double>> rows;

    for (const auto& dataset : kDatasets) {
        const std::string datasetName = dataset.first;
        const std::string inputFile = dataset.second;

        std::vector<std::vector<int>> h2v;
        if (!loadHyperedgeListFromInputFile(inputFile, h2v)) return 1;
        std::vector<std::vector<int>> v2h = vertex2hyperedge(h2v);

        int minVertex = INT_MAX;
        int maxVertex = 0;
        for (const auto& row : h2v) {
            for (int v : row) {
                minVertex = std::min(minVertex, v);
                maxVertex = std::max(maxVertex, v);
            }
        }
        if (minVertex == INT_MAX) minVertex = 1;
        if (maxVertex <= minVertex) maxVertex = std::max(minVertex + 1, kDefaultMaxVertexId);

        for (int incidentSize : kIncidentSizes) {
            HypergraphParams params{};
            params.numHyperedges = static_cast<int>(h2v.size());
            params.minVertexId = minVertex;
            params.maxVertexId = maxVertex;
            params.payloadCapacity = kPayloadCapacity;
            params.alignment = kAlignment;

            HostGraphState initialHostState = buildHostGraphState(h2v, v2h, "");
            DeviceGraphState oldDeviceState =
                constructDeviceGraphState(initialHostState, params.payloadCapacity, params.alignment, params.numHyperedges, params.numHyperedges,
                                          "H2V-old", "V2H-old", "H2H-old");
            DeviceGraphState mutableDeviceState =
                constructDeviceGraphState(initialHostState, params.payloadCapacity, params.alignment, params.numHyperedges, params.numHyperedges,
                                          "H2V-mut", "V2H-mut", "H2H-mut");

            auto t0 = std::chrono::high_resolution_clock::now();
            IncidentUpdateBatch batch = buildIncidentUpdateBatch(params, initialHostState.h2v, incidentSize);
            auto t1 = std::chrono::high_resolution_clock::now();
            applyIncidentVertexUpdates(mutableDeviceState, batch);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<int>> updatedH2V = applyIncidentChangesOnHost(initialHostState.h2v, batch);
            std::vector<std::vector<int>> updatedV2H = vertex2hyperedge(updatedH2V);
            HostGraphState updatedHostState = buildHostGraphState(updatedH2V, updatedV2H, "Updated ");
            int maxId = static_cast<int>(updatedHostState.h2v.size());
            DeviceGraphState updatedDeviceState =
                constructDeviceGraphState(updatedHostState, params.payloadCapacity, params.alignment, maxId, maxId, "H2V-new", "V2H-new", "H2H-new");

            std::vector<int> emptyDeleted;
            std::vector<int> touchedHyperedges = batch.addHyperedgeIds;
            touchedHyperedges.insert(touchedHyperedges.end(), batch.removeHyperedgeIds.begin(), batch.removeHyperedgeIds.end());
            std::sort(touchedHyperedges.begin(), touchedHyperedges.end());
            touchedHyperedges.erase(std::unique(touchedHyperedges.begin(), touchedHyperedges.end()), touchedHyperedges.end());

            std::vector<int> deltaCounts;
            computeMotifCountsDelta(oldDeviceState.h2vOps.context(), oldDeviceState.h2hOps.context(), updatedDeviceState.h2vOps.context(),
                                    updatedDeviceState.h2hOps.context(), emptyDeleted, touchedHyperedges, deltaCounts);
            auto t3 = std::chrono::high_resolution_clock::now();

            double timedMs = std::chrono::duration<double, std::milli>(t1 - t0).count() +
                             std::chrono::duration<double, std::milli>(t2 - t1).count() +
                             std::chrono::duration<double, std::milli>(t3 - t2).count();

            rows.push_back({datasetName, std::to_string(incidentSize / 1000) + "K", timedMs});
            std::cout << datasetName << " incident=" << incidentSize << " timed=" << timedMs << " ms" << std::endl;
        }
    }

    if (!writeOutput(rows)) return 1;
    std::cout << "figure6d benchmark complete." << std::endl;
    return 0;
}
