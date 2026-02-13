#ifndef TEMPORAL_COUNT_HPP
#define TEMPORAL_COUNT_HPP

#include <vector>
#include "temporal_adjacency.hpp"
#include <string>

// Compute 30-bin temporal motif counts for strictly increasing timestamps (Older, Middle, Newest)
// Placeholder signature; Phase 2 will implement the GPU path.
void computeTemporalMotifCountsStrictInc(const TemporalHypergraphIndex& thg,
                                         std::vector<int>& outCounts30);

// Delta for sliding window: counts(newWindow) - counts(oldWindow)
void computeTemporalMotifCountsStrictIncDelta(const TemporalHypergraphIndex& oldWindow,
                                              const TemporalHypergraphIndex& newWindow,
                                              std::vector<int>& outDeltaCounts30);

// Host-side delta format: lines of either
//   A <hyperedgeId> v1 v2 v3 ...
//   R <hyperedgeId> v1 v2 v3 ...
// representing vertices to Add/Remove for that hyperedge in the Newest layer.
// Applies deltas into thg.h2v Newest layer using fill/unfill, and mirrors to v2h.
bool applyTemporalDeltasFromFile(const std::string& path,
                                 TemporalHypergraphIndex& thg);

#endif // TEMPORAL_COUNT_HPP


