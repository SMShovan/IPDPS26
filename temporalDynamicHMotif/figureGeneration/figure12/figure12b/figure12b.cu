#include <algorithm>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
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

struct DatasetStepPercentages {
    std::string dataset;
    double constructionPct = 0.0;
    double deletionPct = 0.0;
    double insertionPct = 0.0;
    double updatePct = 0.0;
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

int nextMultipleOf4(int x) {
    return ((x + 3) / 4) * 4;
}

bool loadHyperedgeListFromSharedInput(const std::string& inputFileName, std::vector<std::vector<int>>& h2v) {
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
            if (j < innerSize) {
                flatValues.push_back(vec2d[i][j]);
            } else if (j == paddedSize - 1) {
                flatValues.push_back(INT_MIN);
            } else {
                flatValues.push_back(0);
            }
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

void applyDeletion(std::vector<std::vector<int>>& h2v, const std::vector<int>& deletedIds) {
    for (int hId : deletedIds) {
        if (hId >= 1 && hId <= static_cast<int>(h2v.size())) {
            h2v[hId - 1].clear();
        }
    }
}

void applyInsertion(std::vector<std::vector<int>>& h2v, const UpdateBatch& batch) {
    int maxAssignedId = static_cast<int>(h2v.size());
    for (int hId : batch.insertAssignedIds) {
        maxAssignedId = std::max(maxAssignedId, hId);
    }
    if (static_cast<int>(h2v.size()) < maxAssignedId) {
        h2v.resize(maxAssignedId);
    }
    for (size_t i = 0; i < batch.generatedInserts.size(); ++i) {
        const int hId = batch.insertAssignedIds[i];
        if (hId >= 1) {
            h2v[hId - 1] = batch.generatedInserts[i];
        }
    }
}

bool writeOutputFile(const std::vector<DatasetStepPercentages>& rows) {
    std::ofstream out("outputStepPercentage.txt");
    if (!out.is_open()) {
        std::cerr << "Error: could not write outputStepPercentage.txt" << std::endl;
        return false;
    }

    out << "Dataset Steps Percentage\n";
    for (const auto& row : rows) {
        out << row.dataset << " Construction " << row.constructionPct << "\n";
        out << row.dataset << " Deletion " << row.deletionPct << "\n";
        out << row.dataset << " Insertion " << row.insertionPct << "\n";
        out << row.dataset << " Update " << row.updatePct << "\n";
    }
    return true;
}

}  // namespace

int main() {
    std::vector<DatasetStepPercentages> rows;
    rows.reserve(kDatasets.size());

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

        double constructionSum = 0.0;
        double deletionSum = 0.0;
        double insertionSum = 0.0;
        double updateSum = 0.0;

        for (int batchSize : kBatchSizes) {
            auto t0 = std::chrono::high_resolution_clock::now();

            TemporalHostState baselineState = buildTemporalHostState(h2v);
            const int baselineNumEdges = static_cast<int>(baselineState.h2v.size());
            const int baselineNumVertices = static_cast<int>(baselineState.v2h.size());

            TemporalHypergraphIndex oldWindow(kDefaultPayloadCapacity, kDefaultAlignment);
            constructLayer(oldWindow, TemporalLayer::Older, baselineState, baselineNumEdges, baselineNumVertices);
            constructLayer(oldWindow, TemporalLayer::Middle, baselineState, baselineNumEdges, baselineNumVertices);
            constructLayer(oldWindow, TemporalLayer::Newest, baselineState, baselineNumEdges, baselineNumVertices);

            TemporalHypergraphIndex newWindow(kDefaultPayloadCapacity, kDefaultAlignment);
            constructLayer(newWindow, TemporalLayer::Older, baselineState, baselineNumEdges, baselineNumVertices);
            constructLayer(newWindow, TemporalLayer::Middle, baselineState, baselineNumEdges, baselineNumVertices);

            auto t1 = std::chrono::high_resolution_clock::now();

            UpdateBatch updateBatch = buildSyntheticUpdateBatch(batchSize, baselineState.h2v,
                                                                maxVerticesPerHyperedge, minVertexId, maxVertexId);
            std::vector<std::vector<int>> updatedH2V = baselineState.h2v;

            auto t2 = std::chrono::high_resolution_clock::now();
            applyDeletion(updatedH2V, updateBatch.deletedIds);
            auto t3 = std::chrono::high_resolution_clock::now();

            applyInsertion(updatedH2V, updateBatch);
            TemporalHostState updatedState = buildTemporalHostState(updatedH2V);
            const int updatedNumEdges = static_cast<int>(updatedState.h2v.size());
            const int updatedNumVertices = static_cast<int>(updatedState.v2h.size());
            constructLayer(newWindow, TemporalLayer::Newest, updatedState, updatedNumEdges, updatedNumVertices);
            auto t4 = std::chrono::high_resolution_clock::now();

            std::vector<int> deltaCounts;
            computeTemporalMotifCountsStrictIncDelta(oldWindow, newWindow, deltaCounts);
            auto t5 = std::chrono::high_resolution_clock::now();

            constructionSum += std::chrono::duration<double>(t1 - t0).count();
            deletionSum += std::chrono::duration<double>(t3 - t2).count();
            insertionSum += std::chrono::duration<double>(t4 - t3).count();
            updateSum += std::chrono::duration<double>(t5 - t4).count();
        }

        const double total = constructionSum + deletionSum + insertionSum + updateSum;
        const double safeTotal = (total > 0.0) ? total : 1.0;

        DatasetStepPercentages row;
        row.dataset = datasetName;
        row.constructionPct = (constructionSum * 100.0) / safeTotal;
        row.deletionPct = (deletionSum * 100.0) / safeTotal;
        row.insertionPct = (insertionSum * 100.0) / safeTotal;
        row.updatePct = (updateSum * 100.0) / safeTotal;
        rows.push_back(row);

        std::cout << "Finished " << datasetName << " -> "
                  << "Construction=" << row.constructionPct
                  << "% Deletion=" << row.deletionPct
                  << "% Insertion=" << row.insertionPct
                  << "% Update=" << row.updatePct
                  << "%" << std::endl;
    }

    if (!writeOutputFile(rows)) {
        return 1;
    }
    std::cout << "figure12b complete. Wrote outputStepPercentage.txt" << std::endl;
    return 0;
}
