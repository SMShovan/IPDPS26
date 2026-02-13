#include "../include/graphGeneration.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>

static inline int type2IntersectionSizeSorted(const std::vector<int>& a, const std::vector<int>& b) {
    int i = 0, j = 0, count = 0;
    while (i < static_cast<int>(a.size()) && j < static_cast<int>(b.size())) {
        if (a[i] == b[j]) { ++count; ++i; ++j; }
        else if (a[i] < b[j]) { ++i; }
        else { ++j; }
    }
    return count;
}

long long type2CountMotifs(const std::vector<std::vector<int>>& h2v,
                           const std::vector<std::vector<int>>& v2h) {
    // Build adjacency of hyperedges to prune pairs to those sharing at least one vertex
    std::vector<std::vector<int>> h2h = hyperedgeAdjacency(v2h, h2v);

    const int E = static_cast<int>(h2v.size());
    long long total = 0;

    for (int i = 0; i < E; ++i) {
        const std::vector<int>& hi = h2v[i];
        for (int neighborOneBased : h2h[i]) {
            int j = neighborOneBased - 1; // adjacency stores 1-based hyperedge IDs
            if (j <= i) continue; // ensure i < j to avoid double counting

            const std::vector<int>& hj = h2v[j];
            int a = type2IntersectionSizeSorted(hi, hj);
            if (a >= 2) {
                int b = static_cast<int>(hi.size()) - a;
                int c = static_cast<int>(hj.size()) - a;
                if (b > 0 && c > 0) {
                    long long aa = a;
                    long long contrib = (aa * (aa - 1) / 2) * static_cast<long long>(b) * static_cast<long long>(c);
                    total += contrib;
                }
            }
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
    std::vector<std::vector<int>> vertexToHyperedge = vertex2hyperedge(hyperedgeToVertex);

    long long motifCount = type2CountMotifs(hyperedgeToVertex, vertexToHyperedge);
    std::cout << "Type2 motif count: " << motifCount << "\n";

    return 0;
}
#endif


