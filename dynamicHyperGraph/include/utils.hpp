#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>

// Structure to hold parsed command line arguments
struct HypergraphParams {
    int numHyperedges;
    int maxVerticesPerHyperedge;
    int minVertexId;
    int maxVertexId;
    int payloadCapacity; // capacity for flattened payload buffers
    int alignment; // padding alignment for payload chunks
    std::string inputFileName; // file name under input/ used to load H2V
    bool saveGenerated; // whether to persist generated H2V under input/
    std::string saveGeneratedFileName; // optional output file name under input/
    int totalChanges; // total number of updates in synthetic batch
    double insertionPercentage; // insertion ratio for synthetic batch (0-100)
    double deletionPercentage; // deletion ratio for synthetic batch (0-100)
};

// Forward declarations
std::pair<std::vector<int>, std::vector<int>> flatten2DVector(const std::vector<std::vector<int>>& vec2d);

// Function declarations
bool parseCommandLineArgs(int argc, char* argv[], HypergraphParams& params);
void printHypergraphParams(const HypergraphParams& params);
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> generateHypergraph(const HypergraphParams& params);
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> loadOrGenerateHypergraph(HypergraphParams& params);
std::vector<std::vector<int>> hyperedgeAdjacency(const std::vector<std::vector<int>>& vertexToHyperedge, const std::vector<std::vector<int>>& hyperedgeToVertex);
std::pair<std::vector<int>, std::vector<int>> flatten(const std::vector<std::vector<int>>& vec2d, const std::string& name);
std::pair<int*, int*> prepareCBSTData(const std::vector<int>& flatIndices);

#endif // UTILS_HPP
