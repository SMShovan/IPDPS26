#include "../include/utils.hpp"
#include "../include/graphGeneration.hpp"
#include "../include/printUtils.hpp"
#include <iostream>
#include <cstdlib>
#include <set>
#include <cstring>

// Function to parse and validate command line arguments
bool parseCommandLineArgs(int argc, char* argv[], HypergraphParams& params) {
    // Check argument count (alignment is optional)
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4] [--temporal] [--temporal-synthetic] [--temporal-deltas=PATH]" << std::endl;
        std::cerr << "Example: " << argv[0] << " 8 5 1 100 4096 8 --temporal --temporal-synthetic --temporal-deltas=updates.txt" << std::endl;
        return false;
    }
    
    // Parse arguments
    params.numHyperedges = std::atoi(argv[1]);
    params.maxVerticesPerHyperedge = std::atoi(argv[2]);
    params.minVertexId = std::atoi(argv[3]);
    params.maxVertexId = std::atoi(argv[4]);
    params.payloadCapacity = std::atoi(argv[5]);
    params.alignment = 4;
    params.enableTemporal = false;
    params.temporalSynthetic = false;
    if (argc >= 7) {
        // Try to parse argv[6] as alignment if numeric, else treat as flags
        char* endptr = nullptr;
        long maybeAlign = std::strtol(argv[6], &endptr, 10);
        int nextArg = 6;
        if (endptr && *endptr == '\0') {
            params.alignment = static_cast<int>(maybeAlign);
            nextArg = 7;
        }
        for (int i = nextArg; i < argc; ++i) {
            std::string a = std::string(argv[i]);
            if (a == "--temporal") params.enableTemporal = true;
            else if (a == "--temporal-synthetic") params.temporalSynthetic = true;
            else if (a.rfind("--temporal-deltas=", 0) == 0) params.temporalDeltasPath = a.substr(strlen("--temporal-deltas="));
        }
    }
    
    // Validate arguments
    if (params.numHyperedges <= 0 || params.maxVerticesPerHyperedge <= 0 || 
        params.minVertexId <= 0 || params.maxVertexId <= 0 || params.payloadCapacity <= 0 || params.alignment <= 0) {
        std::cerr << "Error: All arguments must be positive integers" << std::endl;
        return false;
    }
    
    if (params.minVertexId >= params.maxVertexId) {
        std::cerr << "Error: min_vertex_id must be less than max_vertex_id" << std::endl;
        return false;
    }
    
    return true;
}

// Function to print hypergraph parameters
void printHypergraphParams(const HypergraphParams& params) {
    std::cout << "Generating hypergraph with:" << std::endl;
    std::cout << "  - " << params.numHyperedges << " hyperedges" << std::endl;
    std::cout << "  - Up to " << params.maxVerticesPerHyperedge << " vertices per hyperedge" << std::endl;
    std::cout << "  - Vertex IDs in range [" << params.minVertexId << ", " << params.maxVertexId << "]" << std::endl;
    std::cout << std::endl;
}

// Function to generate hypergraph mappings
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> generateHypergraph(const HypergraphParams& params) {
    // Generate hypergraph data structures
    std::vector<std::vector<int>> hyperedgeToVertex = hyperedge2vertex(params.numHyperedges, params.maxVerticesPerHyperedge, params.minVertexId, params.maxVertexId);
    std::vector<std::vector<int>> vertexToHyperedge = vertex2hyperedge(hyperedgeToVertex);
    
    // Display the generated mappings
    std::cout << "Hyperedge to vertex" << std::endl;
    print2DVector(hyperedgeToVertex);
    std::cout << "Vertex to hyperedge" << std::endl;
    print2DVector(vertexToHyperedge);
    
    return {hyperedgeToVertex, vertexToHyperedge};
}

// Function to generate hyperedge-to-hyperedge adjacency
std::vector<std::vector<int>> hyperedgeAdjacency(const std::vector<std::vector<int>>& vertexToHyperedge, const std::vector<std::vector<int>>& hyperedgeToVertex) {
    int nHyperedges = hyperedgeToVertex.size();
    
    // Resultant adjacency matrix for hyperedges (store 1-based hyperedge IDs)
    std::vector<std::vector<int>> hyperedgeAdjacencyMatrix(nHyperedges);

    // Iterate through each hyperedge (0-indexed)
    for (int hyperedge = 0; hyperedge < nHyperedges; ++hyperedge) {
        std::set<int> adjacentHyperedges;

        // Get the vertices connected by this hyperedge
        const std::vector<int>& vertices = hyperedgeToVertex[hyperedge];

        // For each vertex, find other hyperedges connected to it
        for (int vertex : vertices) {
            for (int otherHyperedge : vertexToHyperedge[vertex]) {
                // vertexToHyperedge stores 1-based hyperedge IDs; avoid self-loop by comparing to (hyperedge + 1)
                if (otherHyperedge != hyperedge + 1) {
                    adjacentHyperedges.insert(otherHyperedge); // Ensure no duplicates
                }
            }
        }

        // Convert set to vector and store in adjacency matrix
        hyperedgeAdjacencyMatrix[hyperedge] = std::vector<int>(adjacentHyperedges.begin(), adjacentHyperedges.end());
    }

    return hyperedgeAdjacencyMatrix;
}

// Function to flatten 2D vector and print debug info
std::pair<std::vector<int>, std::vector<int>> flatten(const std::vector<std::vector<int>>& vec2d, const std::string& name) {
    auto flattened = flatten2DVector(vec2d);
    
    // Print the flattened vectors
    printVector(flattened.first, "Flattened Values (" + name + ")");
    printVector(flattened.second, "Flattened Indices (" + name + ")");
    
    return flattened;
}

// Function to prepare CBST data
std::pair<int*, int*> prepareCBSTData(const std::vector<int>& flatIndices) {
    int* h_values = const_cast<int*>(flatIndices.data());
    int* h_indices = new int[flatIndices.size()];
    for (size_t i = 0; i < flatIndices.size(); ++i) {
        h_indices[i] = i + 1;
    }
    return {h_values, h_indices};
}
