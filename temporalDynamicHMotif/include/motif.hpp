#ifndef MOTIF_HPP
#define MOTIF_HPP

#include "structure.hpp"
#include <vector>

// Runs motif counting on GPU using H2H, V2H, and H2V flattened arrays
// present inside the respective CBST contexts. Prints a 30-bin summary.
// numHyperedges is the number of hyperedges/records (rows) in H2H/H2V.
void computeMotifCounts(const CBSTContext& h2vCtx,
                        const CBSTContext& v2hCtx,
                        const CBSTContext& h2hCtx,
                        int numHyperedges);

#endif // MOTIF_HPP


