#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
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

constexpr int kAlignment = 4;
constexpr int kPayloadCapacity = 1 << 20;
constexpr int kMaxVerticesPerHyperedge = 5;
constexpr int kMinVertexId = 1;
constexpr int kMaxVertexId = 1000000;
constexpr int kTotalChanges = 100000;
const std::vector<int> kHyperedgeCounts = {
    20000000, 25000000, 30000000, 35000000, 40000000, 45000000, 50000000, 55000000
};

int nextMultipleOf4(int value) {
    return ((value + 3) / 4) * 4;
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

DeviceGraphState constructDeviceGraphState(const HostGraphState& hostState, int h2vNumRecords, int h2hNumRecords,
                                           const char* h2vName, const char* v2hName, const char* h2hName) {
    auto h2vPrepared = prepareSingleCBSTInput(hostState.h2vStartOffsets);
    auto v2hPrepared = prepareSingleCBSTInput(hostState.v2hStartOffsets);
    auto h2hPrepared = prepareSingleCBSTInput(hostState.h2hStartOffsets);

    DeviceGraphState deviceState(h2vName, v2hName, h2hName, kPayloadCapacity, kAlignment);
    deviceState.h2vOps.construct(h2vPrepared.keys.get(), h2vPrepared.startOffsets, h2vNumRecords,
                                 const_cast<int*>(hostState.h2vFlatValues.data()), static_cast<int>(hostState.h2vFlatValues.size()));
    deviceState.v2hOps.construct(v2hPrepared.keys.get(), v2hPrepared.startOffsets, static_cast<int>(hostState.v2h.size()),
                                 const_cast<int*>(hostState.v2hFlatValues.data()), static_cast<int>(hostState.v2hFlatValues.size()));
    deviceState.h2hOps.construct(h2hPrepared.keys.get(), h2hPrepared.startOffsets, h2hNumRecords,
                                 const_cast<int*>(hostState.h2hFlatValues.data()), static_cast<int>(hostState.h2hFlatValues.size()));
    return deviceState;
}

UpdateBatch buildSyntheticUpdateBatch(int numHyperedges, const std::vector<std::vector<int>>& originalH2V) {
    HypergraphParams params{};
    params.numHyperedges = numHyperedges;
    params.maxVerticesPerHyperedge = kMaxVerticesPerHyperedge;
    params.minVertexId = kMinVertexId;
    params.maxVertexId = kMaxVertexId;
    params.totalChanges = kTotalChanges;
    params.insertionPercentage = 50.0;
    params.deletionPercentage = 50.0;

    UpdateBatch batch;
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

    batch.generatedInserts = hyperedge2vertex(numInserts, kMaxVerticesPerHyperedge, kMinVertexId, kMaxVertexId);

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

bool writeOutput(const std::vector<std::string>& labels, const std::vector<double>& times) {
    std::ofstream out("outputHyperedgeSize.txt");
    if (!out.is_open()) {
        std::cerr << "Error: cannot write outputHyperedgeSize.txt" << std::endl;
        return false;
    }
    out << "GraphNodes TimeSeconds\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        out << labels[i] << " " << times[i] << "\n";
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
    std::vector<std::string> labels;
    std::vector<double> times;
    labels.reserve(kHyperedgeCounts.size());
    times.reserve(kHyperedgeCounts.size());

    for (int numHyperedges : kHyperedgeCounts) {
        std::cout << "Benchmarking hyperedge count: " << numHyperedges << std::endl;
        std::vector<std::vector<int>> h2v = hyperedge2vertex(numHyperedges, kMaxVerticesPerHyperedge, kMinVertexId, kMaxVertexId);
        std::vector<std::vector<int>> v2h = vertex2hyperedge(h2v);

        HostGraphState initialHostState = buildHostGraphState(h2v, v2h, "");
        DeviceGraphState oldDeviceState = constructDeviceGraphState(initialHostState, numHyperedges, numHyperedges, "H2V-old", "V2H-old", "H2H-old");
        DeviceGraphState mutableDeviceState =
            constructDeviceGraphState(initialHostState, numHyperedges, numHyperedges, "H2V-mut", "V2H-mut", "H2H-mut");

        UpdateBatch updateBatch = buildSyntheticUpdateBatch(numHyperedges, initialHostState.h2v);

        auto t0 = std::chrono::high_resolution_clock::now();
        applyIncrementalUpdates(mutableDeviceState, updateBatch);
        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<int>> updatedH2V = buildUpdatedH2V(initialHostState.h2v, updateBatch);
        std::vector<std::vector<int>> updatedV2H = vertex2hyperedge(updatedH2V);
        HostGraphState updatedHostState = buildHostGraphState(updatedH2V, updatedV2H, "Updated ");
        int maxId = static_cast<int>(updatedHostState.h2v.size());
        DeviceGraphState updatedDeviceState = constructDeviceGraphState(updatedHostState, maxId, maxId, "H2V-new", "V2H-new", "H2H-new");

        std::vector<int> deltaCounts;
        computeMotifCountsDelta(oldDeviceState.h2vOps.context(), oldDeviceState.h2hOps.context(), updatedDeviceState.h2vOps.context(),
                                updatedDeviceState.h2hOps.context(), updateBatch.deletedIds, updateBatch.insertAssignedIds, deltaCounts);
        auto t2 = std::chrono::high_resolution_clock::now();

        double applyTime = std::chrono::duration<double>(t1 - t0).count();
        double deltaTime = std::chrono::duration<double>(t2 - t1).count();
        double total = applyTime + deltaTime;

        labels.push_back(std::to_string(numHyperedges / 1000000) + "M");
        times.push_back(total);
        std::cout << "  timed(apply+delta): " << total << " s" << std::endl;
    }

    if (!writeOutput(labels, times)) return 1;
    std::cout << "figure6b benchmark complete." << std::endl;
    return 0;
}
