#include "../include/graphGeneration.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>

static inline int type3IntersectionSizeSorted(const std::vector<int>& a, const std::vector<int>& b) {
    int i = 0, j = 0, count = 0;
    while (i < static_cast<int>(a.size()) && j < static_cast<int>(b.size())) {
        if (a[i] == b[j]) { ++count; ++i; ++j; }
        else if (a[i] < b[j]) { ++i; }
        else { ++j; }
    }
    return count;
}

static inline int type3TripleIntersectionSizeSorted(const std::vector<int>& a,
                                                    const std::vector<int>& b,
                                                    const std::vector<int>& c) {
    int i = 0, j = 0, k = 0, count = 0;
    while (i < static_cast<int>(a.size()) && j < static_cast<int>(b.size()) && k < static_cast<int>(c.size())) {
        int va = a[i], vb = b[j], vc = c[k];
        if (va == vb && vb == vc) { ++count; ++i; ++j; ++k; }
        else {
            int mn = std::min(va, std::min(vb, vc));
            if (va == mn) ++i;
            if (vb == mn) ++j;
            if (vc == mn) ++k;
        }
    }
    return count;
}

static inline void type3IntersectNeighborsGreaterThanJ(const std::vector<int>& adjI,
                                                       const std::vector<int>& adjJ,
                                                       int jZeroBased,
                                                       std::vector<int>& outCommonKZeroBased) {
    // adj lists are 1-based IDs, sorted ascending
    int p = 0, q = 0;
    outCommonKZeroBased.clear();
    while (p < static_cast<int>(adjI.size()) && q < static_cast<int>(adjJ.size())) {
        int ni = adjI[p];
        int nj = adjJ[q];
        if (ni == nj) {
            int kZero = ni - 1;
            if (kZero > jZeroBased) outCommonKZeroBased.push_back(kZero);
            ++p; ++q;
        } else if (ni < nj) {
            ++p;
        } else {
            ++q;
        }
    }
}

long long type3CountMotifs(const std::vector<std::vector<int>>& h2v,
                           const std::vector<std::vector<int>>& v2h) {
    // Build adjacency (1-based hyperedge IDs per row)
    std::vector<std::vector<int>> h2h = hyperedgeAdjacency(v2h, h2v);
    const int E = static_cast<int>(h2v.size());
    long long total = 0;

    std::vector<int> commonNeighbors;

    for (int i = 0; i < E; ++i) {
        const auto& adjI = h2h[i];
        for (int idOneBased : adjI) {
            int j = idOneBased - 1;
            if (j <= i) continue; // ensure i < j
            const auto& adjJ = h2h[j];
            // Find common neighbors k of i and j with k > j
            type3IntersectNeighborsGreaterThanJ(adjI, adjJ, j, commonNeighbors);
            for (int k : commonNeighbors) {
                // i < j < k guaranteed by filter
                const auto& hi = h2v[i];
                const auto& hj = h2v[j];
                const auto& hk = h2v[k];

                int A = type3IntersectionSizeSorted(hi, hj);
                int B = type3IntersectionSizeSorted(hj, hk);
                int C = type3IntersectionSizeSorted(hi, hk);
                int G = type3TripleIntersectionSizeSorted(hi, hj, hk);

                int x = A - G;
                int y = B - G;
                int z = C - G;
                if (x > 0 && y > 0 && z > 0) {
                    total += static_cast<long long>(x) * static_cast<long long>(y) * static_cast<long long>(z);
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

    // Generate hypergraph mappings
    std::vector<std::vector<int>> hyperedgeToVertex = hyperedge2vertex(
        numHyperedges, maxVerticesPerHyperedge, minVertexId, maxVertexId
    );
    std::vector<std::vector<int>> vertexToHyperedge = vertex2hyperedge(hyperedgeToVertex);

    long long motifCount = type3CountMotifs(hyperedgeToVertex, vertexToHyperedge);
    std::cout << "Type3 motif count: " << motifCount << "\n";

    return 0;
}
#endif


