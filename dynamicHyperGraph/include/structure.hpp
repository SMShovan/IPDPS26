#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

#include <vector>
#include <cuda_runtime.h>

// CBST node definition shared by kernels across translation units
struct CBSTNode {
    int index;
    int value;
    int length;
    int size;
    CBSTNode* left;
    CBSTNode* right;
    CBSTNode* parent;
};

// Device-side context is opaque; host context holds device buffers
struct CBSTContext {
    CBSTNode* d_nodes;
    int* d_keys;
    int* d_startOffsets;
    int* d_flatPayload;
    int* d_insertKeys;
    int* d_insertPayload;
    int* d_insertPrefixSizes;
    int* d_relocationPlan;
    int* d_avail;
    int* d_subtreeAvail;
    int fixedSize;
    int numRecords;
    int initialPayloadSize;
    const char* datasetName;
    int alignment;
};

// Host API for CBST operations (free functions)
void constructCBST(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize, int payloadCapacity, const char* datasetName, CBSTContext& ctx);
void fillCBST(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes, CBSTContext& ctx);
void deleteCBST(const std::vector<int>& deleteKeys, CBSTContext& ctx);
void insertCBST(const std::vector<int>& newKeys, const std::vector<int>& newPayload, const std::vector<int>& newPrefixSizes, CBSTContext& ctx);
void unfillCBST(const std::vector<int>& keysToUnfill, const std::vector<int>& valuesToRemove, const std::vector<int>& removePrefixSizes, CBSTContext& ctx);

// OO wrapper to manage CBST lifecycle and operations
struct CBSTOperations {
    explicit CBSTOperations(const char* datasetName, int payloadCapacity, int alignment = 4);
    ~CBSTOperations();
    CBSTOperations(const CBSTOperations&) = delete;
    CBSTOperations& operator=(const CBSTOperations&) = delete;
    CBSTOperations(CBSTOperations&& other) noexcept;
    CBSTOperations& operator=(CBSTOperations&& other) noexcept;

    void construct(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize);
    void insert(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes);
    void erase(const std::vector<int>& deleteKeys);
    void findAndPrint(const std::vector<int>& ids) const;

    // Accessor to underlying device-resident context (read-only)
    const CBSTContext& context() const;

  private:
    CBSTContext ctx_{};
    bool constructed_ = false;
};

#endif // STRUCTURE_HPP
