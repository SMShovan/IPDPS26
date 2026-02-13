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
#include <utility>
#include <vector>

#include "../../include/graphGeneration.hpp"
#include "../../include/temporal_adjacency.hpp"
#include "../../include/temporal_count.hpp"
#include "../../include/utils.hpp"

namespace fs = std::filesystem;

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

struct DatasetSpec {
    std::string label;
    std::string inputFile;
};

struct TimingRow {
    std::string mode;
    std::string dataset;
    std::string changedEdges;
    double timeSeconds = 0.0;
};

constexpr int kDefaultAlignment = 4;
constexpr int kDefaultPayloadCapacity = 1 << 20;
constexpr int kTHyMeThreads = 8;
constexpr double kTHyMeDelta = 86400000.0;
constexpr int kDeletePercentage = 50;

const std::vector<std::pair<std::string, int>> kChangedEdgeSettings = {
    {"50K", 50000},
    {"100K", 100000},
    {"200K", 200000},
};

const std::vector<DatasetSpec> kDatasets = {
    {"Coauth", "Coauth.txt"},
    {"Tags", "Tags.txt"},
    {"Orkut", "Orkut.txt"},
    {"Threads", "Threads.txt"},
    {"Random", "Random.txt"},
};

int nextMultipleOf4(int x) {
    return ((x + 3) / 4) * 4;
}

bool loadHyperedgeListFromSharedInput(const std::string& inputFileName, std::vector<std::vector<int>>& h2v) {
    const std::string path = std::string("../../../dynamicHyperGraph/input/") + inputFileName;
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
        int vertex = 0;
        while (iss >> vertex) {
            if (vertex <= 0) return false;
            row.push_back(vertex);
        }
        if (!row.empty()) {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            h2v.push_back(std::move(row));
        }
    }
    return !h2v.empty();
}

std::pair<std::vector<int>, std::vector<int>> flatten2DNoPrint(const std::vector<std::vector<int>>& vec2d) {
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

PreparedCBSTData prepareSingleCBSTInput(const std::vector<int>& startOffsets) {
    auto [preparedStartOffsets, preparedKeys] = prepareCBSTData(startOffsets);
    PreparedCBSTData data;
    data.startOffsets = preparedStartOffsets;
    data.keys.reset(preparedKeys);
    return data;
}

TemporalHostState buildTemporalHostState(const std::vector<std::vector<int>>& h2v) {
    TemporalHostState state;
    state.h2v = h2v;
    state.v2h = vertex2hyperedge(state.h2v);
    auto [h2vFlat, h2vStarts] = flatten2DNoPrint(state.h2v);
    auto [v2hFlat, v2hStarts] = flatten2DNoPrint(state.v2h);
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

UpdateBatch buildSyntheticUpdateBatch(const std::vector<std::vector<int>>& originalH2V,
                                      int maxVerticesPerHyperedge,
                                      int minVertexId,
                                      int maxVertexId,
                                      int totalChanges) {
    UpdateBatch batch;
    const int numHyperedges = static_cast<int>(originalH2V.size());
    if (numHyperedges <= 0) return batch;

    const int effectiveChanges = std::min(totalChanges, numHyperedges);
    int numDeletes = static_cast<int>(std::llround(effectiveChanges * (kDeletePercentage / 100.0)));
    int numInserts = effectiveChanges - numDeletes;
    if (numDeletes < 0) numDeletes = 0;
    if (numInserts < 0) numInserts = 0;

    for (int k = 0; k < numDeletes; ++k) batch.deletedIds.push_back(numHyperedges - k);
    std::sort(batch.deletedIds.begin(), batch.deletedIds.end());

    batch.generatedInserts = hyperedge2vertex(numInserts, std::max(1, maxVerticesPerHyperedge),
                                              std::max(1, minVertexId), std::max(minVertexId + 1, maxVertexId));

    const int reuseCount = std::min(numDeletes, numInserts);
    batch.insertAssignedIds.resize(numInserts);
    for (int i = 0; i < reuseCount; ++i) batch.insertAssignedIds[i] = batch.deletedIds[i];
    for (int i = reuseCount; i < numInserts; ++i) batch.insertAssignedIds[i] = numHyperedges + (i - reuseCount) + 1;
    return batch;
}

void applyDeletion(std::vector<std::vector<int>>& h2v, const std::vector<int>& deletedIds) {
    for (int hId : deletedIds) {
        if (hId >= 1 && hId <= static_cast<int>(h2v.size())) h2v[hId - 1].clear();
    }
}

void applyInsertion(std::vector<std::vector<int>>& h2v, const UpdateBatch& batch) {
    int maxAssignedId = static_cast<int>(h2v.size());
    for (int hId : batch.insertAssignedIds) maxAssignedId = std::max(maxAssignedId, hId);
    if (static_cast<int>(h2v.size()) < maxAssignedId) h2v.resize(maxAssignedId);
    for (size_t i = 0; i < batch.generatedInserts.size(); ++i) {
        int hId = batch.insertAssignedIds[i];
        if (hId >= 1) h2v[hId - 1] = batch.generatedInserts[i];
    }
}

double runESCHERTemporal(const std::vector<std::vector<int>>& baseH2V, int totalChanges) {
    int maxVerticesPerHyperedge = 1;
    int minVertexId = INT_MAX;
    int maxVertexId = 0;
    for (const auto& row : baseH2V) {
        maxVerticesPerHyperedge = std::max(maxVerticesPerHyperedge, static_cast<int>(row.size()));
        for (int v : row) {
            minVertexId = std::min(minVertexId, v);
            maxVertexId = std::max(maxVertexId, v);
        }
    }
    if (minVertexId == INT_MAX) minVertexId = 1;
    if (maxVertexId < minVertexId) maxVertexId = minVertexId + 1;

    auto t0 = std::chrono::high_resolution_clock::now();
    TemporalHostState baselineState = buildTemporalHostState(baseH2V);
    int baselineNumEdges = static_cast<int>(baselineState.h2v.size());
    int baselineNumVertices = static_cast<int>(baselineState.v2h.size());

    TemporalHypergraphIndex oldWindow(kDefaultPayloadCapacity, kDefaultAlignment);
    constructLayer(oldWindow, TemporalLayer::Older, baselineState, baselineNumEdges, baselineNumVertices);
    constructLayer(oldWindow, TemporalLayer::Middle, baselineState, baselineNumEdges, baselineNumVertices);
    constructLayer(oldWindow, TemporalLayer::Newest, baselineState, baselineNumEdges, baselineNumVertices);

    TemporalHypergraphIndex newWindow(kDefaultPayloadCapacity, kDefaultAlignment);
    constructLayer(newWindow, TemporalLayer::Older, baselineState, baselineNumEdges, baselineNumVertices);
    constructLayer(newWindow, TemporalLayer::Middle, baselineState, baselineNumEdges, baselineNumVertices);
    auto t1 = std::chrono::high_resolution_clock::now();

    UpdateBatch batch = buildSyntheticUpdateBatch(baseH2V, maxVerticesPerHyperedge, minVertexId, maxVertexId, totalChanges);
    std::vector<std::vector<int>> updatedH2V = baseH2V;
    applyDeletion(updatedH2V, batch.deletedIds);
    applyInsertion(updatedH2V, batch);
    TemporalHostState updatedState = buildTemporalHostState(updatedH2V);
    int updatedNumEdges = static_cast<int>(updatedState.h2v.size());
    int updatedNumVertices = static_cast<int>(updatedState.v2h.size());
    constructLayer(newWindow, TemporalLayer::Newest, updatedState, updatedNumEdges, updatedNumVertices);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<int> deltaCounts;
    computeTemporalMotifCountsStrictIncDelta(oldWindow, newWindow, deltaCounts);
    auto t3 = std::chrono::high_resolution_clock::now();

    double constructSeconds = std::chrono::duration<double>(t1 - t0).count();
    double updateSeconds = std::chrono::duration<double>(t2 - t1).count();
    double deltaSeconds = std::chrono::duration<double>(t3 - t2).count();
    return constructSeconds + updateSeconds + deltaSeconds;
}

bool writeTemporalDatasetForTHyMe(const std::vector<std::vector<int>>& h2v, const std::string& outPath) {
    std::ofstream out(outPath);
    if (!out.is_open()) return false;
    long long ts = 1;
    for (const auto& edge : h2v) {
        if (edge.empty()) continue;
        for (size_t i = 0; i < edge.size(); ++i) {
            if (i) out << ",";
            out << edge[i];
        }
        out << "\t" << ts++ << "\n";
    }
    return true;
}

bool ensureTHyMeParallelBinary() {
    const std::string cmd =
        "cd \"../../../THyMe-main/code\" && g++ -O3 -std=c++11 -fopenmp main_thymeP.cpp -o run_thymeP_omp";
    return std::system(cmd.c_str()) == 0;
}

bool runCommandCapture(const std::string& command, std::string& output) {
    output.clear();
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return false;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
    int code = pclose(pipe);
    return code == 0;
}

bool parseTHyMeRuntime(const std::string& stdoutText, double& seconds) {
    std::istringstream iss(stdoutText);
    std::string tag;
    while (iss >> tag) {
        if (tag == "TOTAL_RUNTIME_SECONDS") {
            if (iss >> seconds) return true;
        }
    }
    return false;
}

double runTHyMeBaseline(const std::vector<std::vector<int>>& baseH2V,
                        int totalChanges,
                        const std::string& datasetLabel,
                        const std::string& changedLabel) {
    int maxVerticesPerHyperedge = 1;
    int minVertexId = INT_MAX;
    int maxVertexId = 0;
    for (const auto& row : baseH2V) {
        maxVerticesPerHyperedge = std::max(maxVerticesPerHyperedge, static_cast<int>(row.size()));
        for (int v : row) {
            minVertexId = std::min(minVertexId, v);
            maxVertexId = std::max(maxVertexId, v);
        }
    }
    if (minVertexId == INT_MAX) minVertexId = 1;
    if (maxVertexId < minVertexId) maxVertexId = minVertexId + 1;

    UpdateBatch batch = buildSyntheticUpdateBatch(baseH2V, maxVerticesPerHyperedge, minVertexId, maxVertexId, totalChanges);
    std::vector<std::vector<int>> updatedH2V = baseH2V;
    applyDeletion(updatedH2V, batch.deletedIds);
    applyInsertion(updatedH2V, batch);

    fs::create_directories("thyme_tmp");
    std::string safeLabel = datasetLabel;
    for (char& c : safeLabel) if (c == '/' || c == ' ') c = '_';
    const std::string graphPath = "thyme_tmp/" + safeLabel + "_" + changedLabel + ".txt";
    const std::string resultPath = "thyme_tmp/" + safeLabel + "_" + changedLabel + "_out.txt";
    if (!writeTemporalDatasetForTHyMe(updatedH2V, graphPath)) return 0.0;

    std::ostringstream cmd;
    cmd << "cd \"../../../THyMe-main/code\" && "
        << "./run_thymeP_omp dummy " << kTHyMeDelta << " " << kTHyMeThreads
        << " \"../../../temporalDynamicHMotif/figureGeneration/figure14/" << graphPath << "\""
        << " \"../../../temporalDynamicHMotif/figureGeneration/figure14/" << resultPath << "\"";

    std::string out;
    if (!runCommandCapture(cmd.str(), out)) return 0.0;
    double runtime = 0.0;
    if (!parseTHyMeRuntime(out, runtime)) {
        std::ifstream in(resultPath);
        if (in.is_open()) in >> runtime;
    }
    return runtime;
}

bool writeTimingTable(const std::vector<TimingRow>& rows) {
    std::ofstream out("escher_vs_thymep_speedup_timing.txt");
    if (!out.is_open()) return false;
    out << "Mode Dataset ChangedEdges TimeSeconds\n";
    for (const auto& r : rows) out << r.mode << " " << r.dataset << " " << r.changedEdges << " " << r.timeSeconds << "\n";
    return true;
}

} // namespace

int main() {
    if (!ensureTHyMeParallelBinary()) {
        std::cerr << "Error: failed to compile THyMe+ OpenMP binary." << std::endl;
        return 1;
    }

    std::vector<TimingRow> rows;
    for (const auto& ds : kDatasets) {
        std::vector<std::vector<int>> h2v;
        if (!loadHyperedgeListFromSharedInput(ds.inputFile, h2v)) return 1;

        for (const auto& setting : kChangedEdgeSettings) {
            const std::string changedLabel = setting.first;
            const int totalChanges = setting.second;

            double thymeSeconds = runTHyMeBaseline(h2v, totalChanges, ds.label, changedLabel);
            rows.push_back({"THyME+", ds.label, changedLabel, thymeSeconds});

            double escherSeconds = runESCHERTemporal(h2v, totalChanges);
            rows.push_back({"ESCHER", ds.label, changedLabel, escherSeconds});

            std::cout << ds.label << " changed=" << changedLabel
                      << " THyME+=" << thymeSeconds
                      << " ESCHER=" << escherSeconds << std::endl;
        }
    }

    if (!writeTimingTable(rows)) {
        std::cerr << "Error: failed to write escher_vs_thymep_speedup_timing.txt" << std::endl;
        return 1;
    }

    std::cout << "figure14 benchmarking complete." << std::endl;
    return 0;
}
