#ifndef MOTIF_UPDATE_HPP
#define MOTIF_UPDATE_HPP

#include "structure.hpp"
#include <vector>

// Computes motif count deltas using frontier-based updates.
// - oldH2vCtx/oldH2hCtx: pre-update contexts (used for deletions)
// - newH2vCtx/newH2hCtx: post-update contexts (used for insertions)
// - deletedHyperedgeIds: 1-based IDs to subtract
// - insertedHyperedgeIds: 1-based IDs to add (includes reused and new IDs)
// - outDeltaCounts: size 30, receives signed deltas (negative for deletions, positive for insertions)
void computeMotifCountsDelta(const CBSTContext& oldH2vCtx,
                             const CBSTContext& oldH2hCtx,
                             const CBSTContext& newH2vCtx,
                             const CBSTContext& newH2hCtx,
                             const std::vector<int>& deletedHyperedgeIds,
                             const std::vector<int>& insertedHyperedgeIds,
                             std::vector<int>& outDeltaCounts);

#endif // MOTIF_UPDATE_HPP


