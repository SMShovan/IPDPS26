#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../include/graphGeneration.hpp"
#include "../../include/utils.hpp"

#define TYPE_MOTIF_LIBRARY
#include "../../src/type1.cu"
#include "../../src/type2.cu"
#include "../../src/type3.cu"

namespace {

struct DatasetSpec {
    std::string label;
    std::string fileName;
};

struct UpdateBatch {
    std::vector<int> deletedIds;
    std::vector<int> insertAssignedIds;
    std::vector<std::vector<int>> insertedHyperedges;
};

struct TypeSpeedupRow {
    std::string dataset;
    std::string changedEdges;
    double type1Speedup = 1.0;
    double type2Speedup = 1.0;
    double type3Speedup = 1.0;
};

struct SubgraphData {
    std::vector<std::vector<int>> h2v;
    std::vector<std::vector<int>> v2h;
};

constexpr double kEpsilonSeconds = 1e-9;
const std::vector<DatasetSpec> kDatasets = {
    {"Coauth", "Coauth.txt"},
    {"Tags", "Tags.txt"},
    {"Orkut", "Orkut.txt"},
    {"Threads", "Threads.txt"},
    {"Random", "Random.txt"},
};
const std::vector<std::string> kChangedEdgeLabels = {"50K", "100K", "200K"};

int parseChangedEdges(const std::string& label) {
    if (label.size() >= 2 && (label.back() == 'K' || label.back() == 'k')) {
        return std::stoi(label.substr(0, label.size() - 1)) * 1000;
    }
    return std::stoi(label);
}

bool loadHyperedgeListFromInputFile(const std::string& inputFileName, std::vector<std::vector<int>>& h2v) {
    const std::string path = std::string("../../input/") + inputFileName;
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
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            h2v.push_back(std::move(row));
        }
    }
    return !h2v.empty();
}

int maxVertexIdInGraph(const std::vector<std::vector<int>>& h2v) {
    int maxVertexId = 1;
    for (const auto& hyperedge : h2v) {
        for (int vertexId : hyperedge) {
            if (vertexId > maxVertexId) {
                maxVertexId = vertexId;
            }
        }
    }
    return maxVertexId;
}

int maxHyperedgeCardinality(const std::vector<std::vector<int>>& h2v) {
    int maxCardinality = 1;
    for (const auto& hyperedge : h2v) {
        maxCardinality = std::max(maxCardinality, static_cast<int>(hyperedge.size()));
    }
    return maxCardinality;
}

UpdateBatch buildSyntheticUpdateBatch(int requestedChanges, const std::vector<std::vector<int>>& originalH2V) {
    UpdateBatch batch;
    const int originalEdgeCount = static_cast<int>(originalH2V.size());
    if (originalEdgeCount <= 0 || requestedChanges <= 0) {
        return batch;
    }

    const int effectiveChanges = std::min(requestedChanges, originalEdgeCount);
    int numDeletes = effectiveChanges / 2;
    int numInserts = effectiveChanges - numDeletes;
    if (numDeletes == 0 && numInserts == 0) {
        numInserts = 1;
    }

    batch.deletedIds.reserve(numDeletes);
    for (int i = 0; i < numDeletes; ++i) {
        batch.deletedIds.push_back(originalEdgeCount - i);
    }
    std::sort(batch.deletedIds.begin(), batch.deletedIds.end());

    const int minVertexId = 1;
    const int maxVertexId = std::max(minVertexId + 1, maxVertexIdInGraph(originalH2V));
    const int maxVerticesPerHyperedge = std::max(1, maxHyperedgeCardinality(originalH2V));
    batch.insertedHyperedges = hyperedge2vertex(numInserts, maxVerticesPerHyperedge, minVertexId, maxVertexId);

    const int reuseCount = std::min(numDeletes, numInserts);
    batch.insertAssignedIds.resize(numInserts);
    for (int i = 0; i < reuseCount; ++i) {
        batch.insertAssignedIds[i] = batch.deletedIds[i];
    }
    for (int i = reuseCount; i < numInserts; ++i) {
        batch.insertAssignedIds[i] = originalEdgeCount + (i - reuseCount) + 1;
    }

    return batch;
}

std::vector<std::vector<int>> buildUpdatedH2V(const std::vector<std::vector<int>>& originalH2V, const UpdateBatch& batch) {
    std::vector<std::vector<int>> updatedH2V = originalH2V;
    int maxAssignedId = static_cast<int>(updatedH2V.size());
    for (int hId : batch.insertAssignedIds) {
        maxAssignedId = std::max(maxAssignedId, hId);
    }
    if (static_cast<int>(updatedH2V.size()) < maxAssignedId) {
        updatedH2V.resize(maxAssignedId);
    }

    for (int deletedId : batch.deletedIds) {
        if (deletedId >= 1 && deletedId <= static_cast<int>(updatedH2V.size())) {
            updatedH2V[deletedId - 1].clear();
        }
    }
    for (size_t i = 0; i < batch.insertAssignedIds.size(); ++i) {
        const int assignedId = batch.insertAssignedIds[i];
        if (assignedId >= 1) {
            updatedH2V[assignedId - 1] = batch.insertedHyperedges[i];
        }
    }
    return updatedH2V;
}

std::vector<int> collectFrontierEdges(const std::vector<std::vector<int>>& h2h, const std::vector<int>& seedIdsOneBased) {
    std::unordered_set<int> selectedZeroBased;
    for (int idOneBased : seedIdsOneBased) {
        const int idZeroBased = idOneBased - 1;
        if (idZeroBased < 0 || idZeroBased >= static_cast<int>(h2h.size())) {
            continue;
        }
        selectedZeroBased.insert(idZeroBased);
        for (int neighborOneBased : h2h[idZeroBased]) {
            const int neighborZeroBased = neighborOneBased - 1;
            if (neighborZeroBased >= 0 && neighborZeroBased < static_cast<int>(h2h.size())) {
                selectedZeroBased.insert(neighborZeroBased);
            }
        }
    }

    std::vector<int> frontier(selectedZeroBased.begin(), selectedZeroBased.end());
    std::sort(frontier.begin(), frontier.end());
    return frontier;
}

SubgraphData buildSubgraph(const std::vector<std::vector<int>>& sourceH2V, const std::vector<int>& selectedEdgesZeroBased) {
    SubgraphData graph;
    graph.h2v.reserve(selectedEdgesZeroBased.size());
    for (int edgeIdx : selectedEdgesZeroBased) {
        if (edgeIdx >= 0 && edgeIdx < static_cast<int>(sourceH2V.size()) && !sourceH2V[edgeIdx].empty()) {
            graph.h2v.push_back(sourceH2V[edgeIdx]);
        }
    }
    graph.v2h = vertex2hyperedge(graph.h2v);
    return graph;
}

template <typename Fn>
double timeInSeconds(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double computeSpeedup(double baselineSeconds, double dynamicSeconds) {
    return baselineSeconds / std::max(dynamicSeconds, kEpsilonSeconds);
}

TypeSpeedupRow benchmarkSingleSetting(const std::string& datasetLabel,
                                      const std::string& changedEdgeLabel,
                                      const std::vector<std::vector<int>>& originalH2V,
                                      const std::vector<std::vector<int>>& originalV2H,
                                      const std::vector<std::vector<int>>& originalH2H) {
    const int requestedChanges = parseChangedEdges(changedEdgeLabel);
    const UpdateBatch batch = buildSyntheticUpdateBatch(requestedChanges, originalH2V);
    const std::vector<std::vector<int>> updatedH2V = buildUpdatedH2V(originalH2V, batch);
    const std::vector<std::vector<int>> updatedV2H = vertex2hyperedge(updatedH2V);
    const std::vector<std::vector<int>> updatedH2H = hyperedgeAdjacency(updatedV2H, updatedH2V);

    const std::vector<int> oldFrontier = collectFrontierEdges(originalH2H, batch.deletedIds);
    const std::vector<int> newFrontier = collectFrontierEdges(updatedH2H, batch.insertAssignedIds);

    volatile long long sink = 0;

    const double type1BaselineSeconds = timeInSeconds([&]() {
        sink ^= type1CountMotifs(updatedH2V);
    });
    const double type1DynamicSeconds = timeInSeconds([&]() {
        const SubgraphData oldSubgraph = buildSubgraph(originalH2V, oldFrontier);
        const SubgraphData newSubgraph = buildSubgraph(updatedH2V, newFrontier);
        sink ^= type1CountMotifs(oldSubgraph.h2v);
        sink ^= type1CountMotifs(newSubgraph.h2v);
    });

    const double type2BaselineSeconds = timeInSeconds([&]() {
        sink ^= type2CountMotifs(updatedH2V, updatedV2H);
    });
    const double type2DynamicSeconds = timeInSeconds([&]() {
        const SubgraphData oldSubgraph = buildSubgraph(originalH2V, oldFrontier);
        const SubgraphData newSubgraph = buildSubgraph(updatedH2V, newFrontier);
        sink ^= type2CountMotifs(oldSubgraph.h2v, oldSubgraph.v2h);
        sink ^= type2CountMotifs(newSubgraph.h2v, newSubgraph.v2h);
    });

    const double type3BaselineSeconds = timeInSeconds([&]() {
        sink ^= type3CountMotifs(updatedH2V, updatedV2H);
    });
    const double type3DynamicSeconds = timeInSeconds([&]() {
        const SubgraphData oldSubgraph = buildSubgraph(originalH2V, oldFrontier);
        const SubgraphData newSubgraph = buildSubgraph(updatedH2V, newFrontier);
        sink ^= type3CountMotifs(oldSubgraph.h2v, oldSubgraph.v2h);
        sink ^= type3CountMotifs(newSubgraph.h2v, newSubgraph.v2h);
    });

    (void)sink;
    TypeSpeedupRow row;
    row.dataset = datasetLabel;
    row.changedEdges = changedEdgeLabel;
    row.type1Speedup = computeSpeedup(type1BaselineSeconds, type1DynamicSeconds);
    row.type2Speedup = computeSpeedup(type2BaselineSeconds, type2DynamicSeconds);
    row.type3Speedup = computeSpeedup(type3BaselineSeconds, type3DynamicSeconds);
    return row;
}

}  // namespace

int main() {
    std::vector<TypeSpeedupRow> rows;
    rows.reserve(kDatasets.size() * kChangedEdgeLabels.size());

    for (const auto& dataset : kDatasets) {
        std::vector<std::vector<int>> originalH2V;
        if (!loadHyperedgeListFromInputFile(dataset.fileName, originalH2V)) {
            return 1;
        }

        const std::vector<std::vector<int>> originalV2H = vertex2hyperedge(originalH2V);
        const std::vector<std::vector<int>> originalH2H = hyperedgeAdjacency(originalV2H, originalH2V);

        for (const std::string& changedEdgeLabel : kChangedEdgeLabels) {
            rows.push_back(benchmarkSingleSetting(
                dataset.label, changedEdgeLabel, originalH2V, originalV2H, originalH2H));
            std::cout << "Benchmarked " << dataset.label << " @ " << changedEdgeLabel << std::endl;
        }
    }

    const std::string outputPath = "figure11_types_speedup.txt";
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open output file: " << outputPath << std::endl;
        return 1;
    }

    out << "Dataset ChangedEdges Type1Speedup Type2Speedup Type3Speedup\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.dataset << " " << row.changedEdges << " "
            << row.type1Speedup << " " << row.type2Speedup << " " << row.type3Speedup << "\n";
    }

    std::cout << "Wrote " << rows.size() << " rows to " << outputPath << std::endl;
    return 0;
}
