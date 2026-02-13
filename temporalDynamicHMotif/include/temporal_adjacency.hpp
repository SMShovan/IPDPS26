#ifndef TEMPORAL_ADJACENCY_HPP
#define TEMPORAL_ADJACENCY_HPP

#include <vector>
#include <string>
#include "temporal_structure.hpp"

// Container grouping H2V and V2H temporal indices
struct TemporalHypergraphIndex {
    TemporalCBSTOperations h2v;
    TemporalCBSTOperations v2h;

    TemporalHypergraphIndex(int payloadCapacity, int alignment = 4)
      : h2v("H2V", payloadCapacity, alignment), v2h("V2H", payloadCapacity, alignment) {}

    void rotate() { h2v.rotate(); v2h.rotate(); }
    void resetNewest() { h2v.resetLayer(TemporalLayer::Newest); v2h.resetLayer(TemporalLayer::Newest); }
};

// Simple CSR-like host representation for adjacency (stub for now)
struct CSRNeighbors {
    int numRows = 0;                 // equals number of anchors
    std::vector<int> rowPtr;         // size numRows + 1
    std::vector<int> colIdx;         // concatenated neighbors (sorted, unique per row)
};

struct CrossAdjacencySpec {
    TemporalLayer anchorLayer;
    TemporalLayer neighborLayer;
    std::vector<int> anchorHyperedgeIds; // 1-based
};

// Build neighbors across layers where H2V[anchor][j] ∩ H2V[neighbor][i] ≠ 0
// Stub: returns empty adjacency (to be implemented with GPU in Phase 2 execution)
CSRNeighbors buildCrossAdjacency(const CrossAdjacencySpec& spec,
                                 const TemporalCBSTOperations& h2v,
                                 const TemporalCBSTOperations& v2h);

// Convenience wrappers for the three pairs we will use
CSRNeighbors buildAdjOlderToMiddle(const std::vector<int>& anchorsMiddle,
                                   const TemporalHypergraphIndex& thg);

CSRNeighbors buildAdjMiddleToNewest(const std::vector<int>& anchorsMiddle,
                                    const TemporalHypergraphIndex& thg);

CSRNeighbors buildAdjOlderToNewest(const std::vector<int>& anchorsOlder,
                                   const TemporalHypergraphIndex& thg);

// Note: definition placed above to resolve forward-use

#endif // TEMPORAL_ADJACENCY_HPP


