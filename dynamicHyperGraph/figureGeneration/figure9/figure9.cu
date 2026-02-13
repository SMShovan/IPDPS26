#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../include/graphGeneration.hpp"
#include "../../include/motif_update.hpp"
#include "../../include/structure.hpp"
#include "../../include/utils.hpp"

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

struct DatasetConfig {
    std::string label;
    std::string inputFile;
};

const std::vector<DatasetConfig> kDatasets = {
    {"Coauth", "Coauth.txt"},
    {"Tags", "Tags.txt"},
    {"Orkut", "Orkut.txt"},
    {"Threads", "Threads.txt"},
    {"Random", "Random.txt"},
};
const std::vector<std::pair<int, std::string>> kChangedEdges = {
    {50000, "50K"},
    {100000, "100K"},
    {200000, "200K"},
};

constexpr int kPayloadCapacity = 1 << 20;
constexpr int kAlignment = 4;
constexpr double kDeletePct = 50.0;
constexpr double kInsertPct = 50.0;
constexpr int kBaselineThreads = 4;

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool loadHypergraphFromInput(const std::string& fileName, std::vector<std::vector<int>>& h2v) {
    const std::string path = std::string("../../input/") + fileName;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error: failed to open input file: " << path << std::endl;
        return false;
    }

    h2v.clear();
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::vector<int> row;
        int v = 0;
        while (iss >> v) {
            if (v <= 0) {
                std::cerr << "Error: non-positive vertex id in " << path << std::endl;
                return false;
            }
            row.push_back(v);
        }
        if (!row.empty()) h2v.push_back(std::move(row));
    }
    if (h2v.empty()) {
        std::cerr << "Error: empty hypergraph in " << path << std::endl;
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

UpdateBatch buildSyntheticUpdateBatch(const HypergraphParams& params, const std::vector<std::vector<int>>& originalH2V) {
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

    if (numDeletes > numHyperedges) numDeletes = numHyperedges;
    if (numDeletes < 0) numDeletes = 0;
    if (numInserts < 0) numInserts = 0;

    for (int k = 0; k < numDeletes; ++k) batch.deletedIds.push_back(numHyperedges - k);
    std::sort(batch.deletedIds.begin(), batch.deletedIds.end());

    int maxVertexId = std::max(params.maxVertexId, params.minVertexId + 1);
    batch.generatedInserts = hyperedge2vertex(numInserts, std::max(1, params.maxVerticesPerHyperedge), std::max(1, params.minVertexId), maxVertexId);

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

std::vector<std::vector<int>> compactNonEmptyHyperedges(const std::vector<std::vector<int>>& h2v) {
    std::vector<std::vector<int>> out;
    out.reserve(h2v.size());
    for (const auto& row : h2v) {
        if (!row.empty()) out.push_back(row);
    }
    return out;
}

bool writeMoCHyCsv(const std::vector<std::vector<int>>& h2v, const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        std::cerr << "Error: failed to open csv output: " << outputPath << std::endl;
        return false;
    }
    for (const auto& row : h2v) {
        if (row.empty()) continue;
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) out << ",";
            out << std::max(0, row[i] - 1);
        }
        out << "\n";
    }
    return true;
}

bool ensureMoCHyExecutable(const std::string& executablePath) {
    if (std::filesystem::exists(executablePath)) return true;
    const std::string compileCmd =
        "g++ -O3 -std=c++11 -fopenmp \"../../../MoCHy-master/main_exact_par.cpp\" -o \"" + executablePath + "\"";
    std::cout << "Compiling Baseline executable..." << std::endl;
    int rc = std::system(compileCmd.c_str());
    return rc == 0 && std::filesystem::exists(executablePath);
}

bool runBaselineAndParseRuntime(const std::string& executablePath, const std::string& csvInputPath, double& runtimeSeconds) {
    const std::string cmd = "\"" + executablePath + "\" " + std::to_string(kBaselineThreads) + " \"" + csvInputPath + "\"";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[4096];
    bool found = false;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line = trim(buffer);
        const std::string prefix = "TOTAL_RUNTIME_SECONDS";
        if (line.rfind(prefix, 0) == 0) {
            std::istringstream iss(line.substr(prefix.size()));
            iss >> runtimeSeconds;
            found = !iss.fail();
        }
    }

    int rc = pclose(pipe);
    return rc == 0 && found;
}

bool writeOutput(const std::vector<std::tuple<std::string, std::string, std::string, double>>& rows) {
    std::ofstream out("speedup_timing.txt");
    if (!out.is_open()) {
        std::cerr << "Error: cannot write output file" << std::endl;
        return false;
    }
    out << "Mode Dataset ChangedEdges TimeSeconds\n";
    for (const auto& row : rows) {
        out << std::get<0>(row) << " " << std::get<1>(row) << " " << std::get<2>(row) << " " << std::get<3>(row) << "\n";
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
        int paddedSize = ((innerSize + 3) / 4) * 4;
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
    const std::string baselineExe = "../../../MoCHy-master/exact_cli_par";
    if (!ensureMoCHyExecutable(baselineExe)) {
        std::cerr << "Failed to compile/find baseline executable at " << baselineExe << std::endl;
        return 1;
    }

    std::vector<std::tuple<std::string, std::string, std::string, double>> rows;

    for (const auto& ds : kDatasets) {
        std::vector<std::vector<int>> h2v;
        if (!loadHypergraphFromInput(ds.inputFile, h2v)) return 1;
        auto v2h = vertex2hyperedge(h2v);

        int maxVerticesPerHyperedge = 1;
        int minVertex = INT_MAX;
        int maxVertex = 0;
        for (const auto& row : h2v) {
            maxVerticesPerHyperedge = std::max(maxVerticesPerHyperedge, static_cast<int>(row.size()));
            for (int v : row) {
                minVertex = std::min(minVertex, v);
                maxVertex = std::max(maxVertex, v);
            }
        }
        if (minVertex == INT_MAX) minVertex = 1;
        if (maxVertex <= minVertex) maxVertex = minVertex + 1;

        for (const auto& changed : kChangedEdges) {
            int requestedTotalChanges = changed.first;
            int effectiveTotalChanges = std::min(requestedTotalChanges, std::max(1, 2 * static_cast<int>(h2v.size())));

            HypergraphParams params{};
            params.numHyperedges = static_cast<int>(h2v.size());
            params.maxVerticesPerHyperedge = maxVerticesPerHyperedge;
            params.minVertexId = minVertex;
            params.maxVertexId = maxVertex;
            params.payloadCapacity = kPayloadCapacity;
            params.alignment = kAlignment;
            params.totalChanges = effectiveTotalChanges;
            params.deletionPercentage = kDeletePct;
            params.insertionPercentage = kInsertPct;

            HostGraphState initialHost = buildHostGraphState(h2v, v2h, "");
            DeviceGraphState oldDevice = constructDeviceGraphState(initialHost, params.numHyperedges, params.numHyperedges, "H2V-old", "V2H-old", "H2H-old");
            DeviceGraphState mutableDevice =
                constructDeviceGraphState(initialHost, params.numHyperedges, params.numHyperedges, "H2V-mut", "V2H-mut", "H2H-mut");

            auto t0 = std::chrono::high_resolution_clock::now();
            UpdateBatch batch = buildSyntheticUpdateBatch(params, initialHost.h2v);
            auto t1 = std::chrono::high_resolution_clock::now();
            applyIncrementalUpdates(mutableDevice, batch);
            auto t2 = std::chrono::high_resolution_clock::now();

            auto updatedH2V = buildUpdatedH2V(initialHost.h2v, batch);
            auto updatedV2H = vertex2hyperedge(updatedH2V);
            HostGraphState updatedHost = buildHostGraphState(updatedH2V, updatedV2H, "Updated ");
            int maxId = static_cast<int>(updatedHost.h2v.size());
            DeviceGraphState newDevice = constructDeviceGraphState(updatedHost, maxId, maxId, "H2V-new", "V2H-new", "H2H-new");

            std::vector<int> deltaCounts;
            computeMotifCountsDelta(oldDevice.h2vOps.context(), oldDevice.h2hOps.context(), newDevice.h2vOps.context(), newDevice.h2hOps.context(),
                                    batch.deletedIds, batch.insertAssignedIds, deltaCounts);
            auto t3 = std::chrono::high_resolution_clock::now();

            double escherSeconds = std::chrono::duration<double>(t1 - t0).count() + std::chrono::duration<double>(t2 - t1).count() +
                                   std::chrono::duration<double>(t3 - t2).count();
            rows.push_back({"ESCHER", ds.label, changed.second, escherSeconds});

            auto compactChanged = compactNonEmptyHyperedges(updatedH2V);
            const std::string csvPath = "tmp_" + ds.label + "_" + changed.second + ".csv";
            if (!writeMoCHyCsv(compactChanged, csvPath)) return 1;

            double baselineSeconds = 0.0;
            if (!runBaselineAndParseRuntime(baselineExe, csvPath, baselineSeconds)) {
                std::cerr << "Failed to run baseline for " << ds.label << " changed=" << changed.second << std::endl;
                return 1;
            }
            rows.push_back({"Baseline", ds.label, changed.second, baselineSeconds});
        }
    }

    if (!writeOutput(rows)) return 1;
    std::cout << "figure9 benchmark complete." << std::endl;
    return 0;
}
