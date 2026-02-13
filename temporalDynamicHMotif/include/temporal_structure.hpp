#ifndef TEMPORAL_STRUCTURE_HPP
#define TEMPORAL_STRUCTURE_HPP

#include <string>
#include <vector>
#include <memory>

// Forward declarations to avoid pulling in CUDA headers here
struct CBSTContext;
struct CBSTOperations;
void unfillCBST(const std::vector<int>& keysToUnfill,
                const std::vector<int>& valuesToRemove,
                const std::vector<int>& removePrefixSizes,
                CBSTContext& ctx);
void fillCBST(const std::vector<int>& keysToFill,
              const std::vector<int>& valuesToInsert,
              const std::vector<int>& insertPrefixSizes,
              CBSTContext& ctx);

// Temporal layer identifiers for a 3-slot ring buffer
enum class TemporalLayer { Older = 0, Middle = 1, Newest = 2 };

// Wrapper that manages three CBSTOperations instances as a temporal ring buffer
class TemporalCBSTOperations {
  public:
    TemporalCBSTOperations(const std::string& label,
                           int payloadCapacity,
                           int alignment = 4);
    ~TemporalCBSTOperations();

    void rotate();
    void resetLayer(TemporalLayer layer);

    void construct(TemporalLayer layer,
                   int* cbstKeys,
                   int* cbstStartOffsets,
                   int numRecords,
                   int* flatPayload,
                   int flatPayloadSize);

    void insert(TemporalLayer layer,
                const std::vector<int>& keys,
                const std::vector<int>& payload,
                const std::vector<int>& prefix);

    void erase(TemporalLayer layer, const std::vector<int>& keys);

    void fill(TemporalLayer layer,
              const std::vector<int>& keys,
              const std::vector<int>& payload,
              const std::vector<int>& prefix);

    void unfill(TemporalLayer layer,
                const std::vector<int>& keys,
                const std::vector<int>& valuesToRemove,
                const std::vector<int>& removePrefix);

    const CBSTContext& context(TemporalLayer layer) const;
    CBSTContext& context(TemporalLayer layer);

    TemporalLayer oldest() const;
    TemporalLayer middle() const;
    TemporalLayer newest() const;

  private:
    int resolveIndex(TemporalLayer layer) const;
    const CBSTOperations& slot(TemporalLayer layer) const;
    CBSTOperations& slot(TemporalLayer layer);
    CBSTContext& mutableContext(TemporalLayer layer);

  private:
    std::string label_;
    int payloadCapacity_;
    int alignment_;
    int oldestIdx_ = 0; // 0..2, indicates which physical slot is logically Older
    std::string slotLabels_[3];
    std::unique_ptr<CBSTOperations> slots_[3];
};

#endif // TEMPORAL_STRUCTURE_HPP


