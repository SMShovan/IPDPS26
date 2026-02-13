#include "../include/graphGeneration.hpp"
#include "../include/printUtils.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <set>

std::vector<std::vector<int>> hyperedge2vertex(int n, int m, int r1, int r2) {
    std::vector<std::vector<int>> vec2d(n);
    std::srand(std::time(0)); // Seed for random number generation

    for (int i = 0; i < n; ++i) {
        int innerSize = rand() % m + 1; // Random inner size from 1 to m
        std::set<int> uniqueVertices; // Use set to ensure uniqueness
        
        // Generate unique vertex IDs for this hyperedge (1-indexed)
        while (uniqueVertices.size() < static_cast<size_t>(innerSize)) {
            int vertexId = rand() % (r2 - r1 + 1) + r1; // Keep 1-based IDs to avoid 0 sentinel
            uniqueVertices.insert(vertexId);
        }
        
        // Convert set to vector
        vec2d[i] = std::vector<int>(uniqueVertices.begin(), uniqueVertices.end());
    }

    return vec2d;
}

std::vector<std::vector<int>> vertex2hyperedge(const std::vector<std::vector<int>>& hyperedgeToVertex) {
    // Step 1: Find the maximum value in hyperedgeToVertex
    int maxValue = 0;
    for (const auto& row : hyperedgeToVertex) {
        if (!row.empty()) {
            maxValue = std::max(maxValue, *std::max_element(row.begin(), row.end()));
        }
    }

    // Step 2: Initialize vertexToHyperedge with size maxValue + 1 (index 0 unused; 1-based vertex IDs)
    std::vector<std::vector<int>> vertexToHyperedge(maxValue + 1);

    // Step 3: Populate vertexToHyperedge with indices from hyperedgeToVertex
    for (size_t rowIndex = 0; rowIndex < hyperedgeToVertex.size(); ++rowIndex) {
        for (int value : hyperedgeToVertex[rowIndex]) {
            vertexToHyperedge[value].push_back(static_cast<int>(rowIndex) + 1);  // store 1-based hyperedge IDs
        }
    }

    return vertexToHyperedge;
}

