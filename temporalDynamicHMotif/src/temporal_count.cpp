#include "../include/temporal_count.hpp"

void computeTemporalMotifCountsStrictInc(const TemporalHypergraphIndex& thg,
                                         std::vector<int>& outCounts30) {
    (void)thg;
    outCounts30.assign(30, 0);
    // Phase 2: implement GPU enumeration and fill counts.
}

void computeTemporalMotifCountsStrictIncDelta(const TemporalHypergraphIndex& thg,
                                              const std::vector<int>& evictedMiddleAnchors,
                                              const std::vector<int>& newMiddleAnchors,
                                              std::vector<int>& outDeltaCounts30) {
    (void)thg; (void)evictedMiddleAnchors; (void)newMiddleAnchors;
    outDeltaCounts30.assign(30, 0);
    // Phase 3: implement subtract (old window) and add (new window).
}


