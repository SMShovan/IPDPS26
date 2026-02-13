#include "../include/graphGeneration.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>

long long type1CountMotifs(const std::vector<std::vector<int>>& hyperedgeToVertex) {
    long long total = 0;
    for (const auto& hyperedge : hyperedgeToVertex) {
        const long long n = static_cast<long long>(hyperedge.size());
        if (n >= 3) {
            total += (n * (n - 1) * (n - 2)) / 6;
        }
    }
    return total;
}

#ifndef TYPE_MOTIF_LIBRARY
int main(int argc, char* argv[]) {
    // Parameters: <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id>
    int numHyperedges = 10;
    int maxVerticesPerHyperedge = 5;
    int minVertexId = 1;
    int maxVertexId = 50;

    if (argc == 5) {
        numHyperedges = std::atoi(argv[1]);
        maxVerticesPerHyperedge = std::atoi(argv[2]);
        minVertexId = std::atoi(argv[3]);
        maxVertexId = std::atoi(argv[4]);
    } else {
        std::cout << "Usage: " << argv[0]
                  << " <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id>\n";
        std::cout << "Using defaults: "
                  << numHyperedges << " "
                  << maxVerticesPerHyperedge << " "
                  << minVertexId << " "
                  << maxVertexId << "\n";
    }

    // Generate hypergraph (1-based vertex IDs, no duplicates per hyperedge, size in [1, m])
    std::vector<std::vector<int>> hyperedgeToVertex = hyperedge2vertex(
        numHyperedges, maxVerticesPerHyperedge, minVertexId, maxVertexId
    );

    // Optional: print the generated incidence
    // print2DVector(hyperedgeToVertex);

    // Count type1 motifs: sum over hyperedges of C(n, 3)
    long long motifCount = type1CountMotifs(hyperedgeToVertex);
    std::cout << "Type1 motif count (sum_e C(|e|, 3)): " << motifCount << "\n";

    return 0;
}
#endif

