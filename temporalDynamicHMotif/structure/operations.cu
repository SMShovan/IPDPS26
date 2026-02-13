#include "../include/structure.hpp"
#include "../include/utils.hpp"
#include "../include/printUtils.hpp"
#include <iostream>
#include <cstdlib>
#include <climits>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// Kernel prototypes moved to kernel/kernels.cuh
#include "../kernel/kernels.cuh"

// Utility: dump node index and value to plain arrays
__global__ void dumpNodeIndexValue(CBSTNode* nodes, int n, int* outIndex, int* outValue) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        outIndex[tid] = nodes[tid].index;
        outValue[tid] = nodes[tid].value;
    }
}

// Local CUDA error checker for this TU
static inline void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

void constructCBST(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize, int payloadCapacity, const char* datasetName, CBSTContext& ctx) {
    ctx.fixedSize = payloadCapacity;
    ctx.numRecords = numRecords;
    ctx.initialPayloadSize = flatPayloadSize;
    ctx.datasetName = datasetName;
    // ctx.alignment should be set by the owner (CBSTOperations) before calling

    if (numRecords == 0) {
        return;
    }

    if (ctx.fixedSize < flatPayloadSize) {
        std::cerr << "Overflow: fixedSize is less than flatPayloadSize" << std::endl;
        return;
    }

    checkCuda(cudaMalloc(&ctx.d_nodes, numRecords * sizeof(CBSTNode)));
    checkCuda(cudaMalloc(&ctx.d_keys, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_startOffsets, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_flatPayload, ctx.fixedSize * sizeof(int)));

    checkCuda(cudaMemcpy(ctx.d_keys, keys, numRecords * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_startOffsets, startOffsets, numRecords * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(ctx.d_flatPayload, flatPayload, flatPayloadSize * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(ctx.d_flatPayload + flatPayloadSize, 0, (ctx.fixedSize - flatPayloadSize) * sizeof(int)));

    checkCuda(cudaMalloc(&ctx.d_insertKeys, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_insertPayload, numRecords * 3 * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_insertPrefixSizes, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_relocationPlan, 3 * numRecords * sizeof(int)));
    // Availability arrays (0/1 per node) and subtree sums (per node)
    checkCuda(cudaMalloc(&ctx.d_avail, numRecords * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_subtreeAvail, numRecords * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_avail, 0, numRecords * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_subtreeAvail, 0, numRecords * sizeof(int)));

    int blockSize = 256;
    int numBlocks = (numRecords + blockSize - 1) / blockSize;

    buildEmptyBinaryTree<<<numBlocks, blockSize>>>(ctx.d_nodes, numRecords);
    checkCuda(cudaDeviceSynchronize());

    storeItemsIntoNodes<<<numBlocks, blockSize>>>(ctx.d_nodes, ctx.d_keys, ctx.d_startOffsets, numRecords, flatPayloadSize);
    checkCuda(cudaDeviceSynchronize());

    std::cout << "Printing the tree from the device (" << datasetName << "):" << std::endl;
    printEachNode<<<numBlocks, blockSize>>>(ctx.d_nodes, numRecords);
    checkCuda(cudaDeviceSynchronize());
}

void fillCBST(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes, CBSTContext& ctx) {
    if (insertKeys.empty()) return;
    std::vector<int> relocationPlanHost(insertKeys.size() * 3, 0);

    checkCuda(cudaMemcpy(ctx.d_insertKeys, insertKeys.data(), insertKeys.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPayload, insertPayload.data(), insertPayload.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, insertPrefixSizes.data(), insertPrefixSizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_relocationPlan, relocationPlanHost.data(), relocationPlanHost.size() * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (static_cast<int>(insertKeys.size()) + blockSize - 1) / blockSize;
    insertNode<<<numBlocks, blockSize>>>(ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes, static_cast<int>(insertKeys.size()), ctx.d_relocationPlan);
    checkCuda(cudaDeviceSynchronize());

    int K = static_cast<int>(insertKeys.size());
    int* d_tmp;
    checkCuda(cudaMalloc(&d_tmp, K * sizeof(int)));
    // Note: currently kernel pads to 4. To support arbitrary alignment, update the kernel to accept ctx.alignment.
    computeNextMultipleOf4<<<(K + blockSize - 1) / blockSize, blockSize>>>(ctx.d_relocationPlan, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());
    thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(d_tmp);
    thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
    checkCuda(cudaDeviceSynchronize());
    updatePartialSolution<<<(K + blockSize - 1) / blockSize, blockSize>>>(ctx.d_relocationPlan, d_tmp, K);
    checkCuda(cudaDeviceSynchronize());

    std::vector<int> relocationPlanHostOut(K * 3);
    checkCuda(cudaMemcpy(relocationPlanHostOut.data(), ctx.d_relocationPlan, K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(relocationPlanHostOut, "Cumulative Relocation Plan");

    printf("[%s] Space available from: %d \n", ctx.datasetName, ctx.initialPayloadSize);
    allocateSpace<<<numBlocks, blockSize>>>(ctx.d_relocationPlan, ctx.d_flatPayload, ctx.initialPayloadSize, ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes, K);
    checkCuda(cudaDeviceSynchronize());

    // Advance appended-space cursor so next batch appends after this one
    if (K > 0) {
        int totalAppended = relocationPlanHostOut[3 * (K - 1) + 2];
        ctx.initialPayloadSize += totalAppended;
    }

    std::vector<int> updatedFlat(ctx.fixedSize);
    checkCuda(cudaMemcpy(updatedFlat.data(), ctx.d_flatPayload, ctx.fixedSize * sizeof(int), cudaMemcpyDeviceToHost));
    printVector(updatedFlat, "Updated Flattened Values (vec1d)");

    checkCuda(cudaFree(d_tmp));
}

void deleteCBST(const std::vector<int>& deleteKeys, CBSTContext& ctx) {
    if (deleteKeys.empty()) return;
    int* d_deleteKeys;
    checkCuda(cudaMalloc(&d_deleteKeys, deleteKeys.size() * sizeof(int)));
    checkCuda(cudaMemcpy(d_deleteKeys, deleteKeys.data(), deleteKeys.size() * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (static_cast<int>(deleteKeys.size()) + blockSize - 1) / blockSize;
    // Mark node index = -1 (lazy delete)
    deleteNode<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deleteKeys, static_cast<int>(deleteKeys.size()));
    checkCuda(cudaDeviceSynchronize());

    // Mark avail = 1 for deleted nodes
    markAvail<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deleteKeys, static_cast<int>(deleteKeys.size()), ctx.d_avail);
    checkCuda(cudaDeviceSynchronize());

    // Bottom-up level-wise reduction
    // Compute last level start using heap property
    int lastLevelStart = 1;
    while (lastLevelStart * 2 <= ctx.numRecords) lastLevelStart <<= 1; // 2^h
    int levelStart = lastLevelStart - 1; // 0-based index of first node at last full level
    if (levelStart >= ctx.numRecords) levelStart = (lastLevelStart >> 1) - 1; // adjust if beyond size

    // Initialize subtreeAvail = avail at leaves and beyond
    // For ranges beyond numRecords, threads will just skip
    // Propagate from bottom to top
    int currStart = ctx.numRecords - 1; // last index
    // First, copy avail to subtreeAvail for all nodes
    int nodesBlocks = (ctx.numRecords + blockSize - 1) / blockSize;
    reduceAvailLevel<<<nodesBlocks, blockSize>>>(ctx.numRecords - 1, ctx.numRecords - 1, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
    checkCuda(cudaDeviceSynchronize());

    // Now perform level-wise reduction
    for (int levelEnd = ctx.numRecords - 1; levelStart >= 0; levelStart = (levelStart - 1) / 2) {
        int start = levelStart;
        int end = levelEnd;
        int count = end - start + 1;
        int blocks = (count + blockSize - 1) / blockSize;
        reduceAvailLevel<<<blocks, blockSize>>>(start, end, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
        checkCuda(cudaDeviceSynchronize());
        if (levelStart == 0) break;
        levelEnd = levelStart - 1;
        levelStart = (levelStart - 1) / 2;
    }

    checkCuda(cudaFree(d_deleteKeys));
}

void insertCBST(const std::vector<int>& newKeys,
                const std::vector<int>& newPayload,
                const std::vector<int>& newPrefixSizes,
                CBSTContext& ctx) {
    if (newKeys.empty()) return;
    // Copy inputs to device (reuse insert buffers)
    checkCuda(cudaMemcpy(ctx.d_insertKeys, newKeys.data(), newKeys.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPayload, newPayload.data(), newPayload.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, newPrefixSizes.data(), newPrefixSizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    // zero relocation plan
    std::vector<int> zeroReloc(newKeys.size() * 3, 0);
    checkCuda(cudaMemcpy(ctx.d_relocationPlan, zeroReloc.data(), zeroReloc.size() * sizeof(int), cudaMemcpyHostToDevice));
    // Determine number of deleted nodes available (root's subtreeAvail)
    int deletedCountHost = 0;
    checkCuda(cudaMemcpy(&deletedCountHost, ctx.d_subtreeAvail, sizeof(int), cudaMemcpyDeviceToHost));
    // Launch kernel for reuse up to min(K, deletedCount)
    int K = static_cast<int>(newKeys.size());
    int reuseK = (deletedCountHost < K) ? deletedCountHost : K;
    int blockSize = 256;
    int numBlocksReuse = (reuseK + blockSize - 1) / blockSize;
    if (reuseK > 0) {
        insertIntoDeletedKth<<<numBlocksReuse, blockSize>>>(ctx.d_nodes,
                                                   ctx.d_flatPayload,
                                                   ctx.d_subtreeAvail,
                                                   ctx.d_avail,
                                                   ctx.numRecords,
                                                   ctx.d_insertKeys,
                                                   ctx.d_insertPayload,
                                                   ctx.d_insertPrefixSizes,
                                                   ctx.d_relocationPlan,
                                                   reuseK);
        checkCuda(cudaDeviceSynchronize());
    }
    // Handle any overflows via relocation plan (reuse fillCBST pipeline)
    if (reuseK > 0) {
        int* d_tmp;
        checkCuda(cudaMalloc(&d_tmp, reuseK * sizeof(int)));
        computeNextMultipleOf4<<<(reuseK + blockSize - 1) / blockSize, blockSize>>>(ctx.d_relocationPlan, d_tmp, reuseK);
        checkCuda(cudaDeviceSynchronize());
        thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(d_tmp);
        thrust::inclusive_scan(tmp_ptr, tmp_ptr + reuseK, tmp_ptr);
        checkCuda(cudaDeviceSynchronize());
        updatePartialSolution<<<(reuseK + blockSize - 1) / blockSize, blockSize>>>(ctx.d_relocationPlan, d_tmp, reuseK);
        checkCuda(cudaDeviceSynchronize());
        // Read relocation summary to bump tail
        std::vector<int> relocationPlanHostOut(reuseK * 3);
        checkCuda(cudaMemcpy(relocationPlanHostOut.data(), ctx.d_relocationPlan, reuseK * 3 * sizeof(int), cudaMemcpyDeviceToHost));
        allocateSpace<<<numBlocksReuse, blockSize>>>(ctx.d_relocationPlan, ctx.d_flatPayload, ctx.initialPayloadSize, ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes, reuseK);
        checkCuda(cudaDeviceSynchronize());
        if (reuseK > 0) {
            int totalAppended = relocationPlanHostOut[3 * (reuseK - 1) + 2];
            ctx.initialPayloadSize += totalAppended;
        }
        checkCuda(cudaFree(d_tmp));
    }

    // Surplus inserts beyond deleted slots: append at tail, then reconstruct
    int surplus = K - reuseK;
    if (surplus > 0) {
        // Helper for alignment
        auto nextMultiple = [](int num, int a) {
            if (num <= 0) return 0;
            int q = (num + a - 1) / a;
            return q * a;
        };
        std::vector<int> appendedOffsets;
        appendedOffsets.reserve(surplus);
        int cursor = ctx.initialPayloadSize;
        for (int i = 0; i < surplus; ++i) {
            int globalIdx = reuseK + i;
            int start = (globalIdx == 0) ? 0 : newPrefixSizes[globalIdx - 1];
            int end = newPrefixSizes[globalIdx];
            int len = end - start;
            int aligned = nextMultiple(len, ctx.alignment);
            int base = cursor;
            appendedOffsets.push_back(base);
            // copy payload
            if (len > 0) {
                checkCuda(cudaMemcpy(ctx.d_flatPayload + base, newPayload.data() + start, len * sizeof(int), cudaMemcpyHostToDevice));
            }
            int sentinel = INT_MIN;
            checkCuda(cudaMemcpy(ctx.d_flatPayload + base + aligned, &sentinel, sizeof(int), cudaMemcpyHostToDevice));
            cursor += aligned + 1; // include sentinel slot
        }
        ctx.initialPayloadSize = cursor;

        // Reconstruct CBST with N' = N + surplus
        int oldN = ctx.numRecords;
        int newN = oldN + surplus;

        // Dump existing (index,value) from device
        int *d_idx, *d_val;
        checkCuda(cudaMalloc(&d_idx, oldN * sizeof(int)));
        checkCuda(cudaMalloc(&d_val, oldN * sizeof(int)));
        int blocksDump = (oldN + blockSize - 1) / blockSize;
        dumpNodeIndexValue<<<blocksDump, blockSize>>>(ctx.d_nodes, oldN, d_idx, d_val);
        checkCuda(cudaDeviceSynchronize());
        std::vector<int> h_idx(oldN), h_val(oldN);
        checkCuda(cudaMemcpy(h_idx.data(), d_idx, oldN * sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(h_val.data(), d_val, oldN * sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaFree(d_idx));
        checkCuda(cudaFree(d_val));
        // Pair and sort by index ascending
        std::vector<std::pair<int,int>> pairs;
        pairs.reserve(oldN);
        for (int i = 0; i < oldN; ++i) {
            if (h_idx[i] > 0) pairs.emplace_back(h_idx[i], h_val[i]);
        }
        std::sort(pairs.begin(), pairs.end(), [](const std::pair<int,int>& a, const std::pair<int,int>& b){ return a.first < b.first; });

        // Build new host arrays
        std::vector<int> h_newKeys(newN);
        std::vector<int> h_newStarts(newN);
        for (int i = 0; i < newN; ++i) h_newKeys[i] = i + 1;
        // existing
        for (int i = 0; i < oldN && i < static_cast<int>(pairs.size()); ++i) {
            h_newStarts[i] = pairs[i].second;
        }
        // appended
        for (int i = 0; i < surplus; ++i) {
            h_newStarts[oldN + i] = appendedOffsets[i];
        }

        // Rebuild nodes and key/start arrays (without touching flat payload)
        // Free old arrays
        if (ctx.d_keys) checkCuda(cudaFree(ctx.d_keys));
        if (ctx.d_startOffsets) checkCuda(cudaFree(ctx.d_startOffsets));
        if (ctx.d_nodes) checkCuda(cudaFree(ctx.d_nodes));
        if (ctx.d_avail) checkCuda(cudaFree(ctx.d_avail));
        if (ctx.d_subtreeAvail) checkCuda(cudaFree(ctx.d_subtreeAvail));
        // Resize insert buffers as well
        if (ctx.d_insertKeys) checkCuda(cudaFree(ctx.d_insertKeys));
        if (ctx.d_insertPayload) checkCuda(cudaFree(ctx.d_insertPayload));
        if (ctx.d_insertPrefixSizes) checkCuda(cudaFree(ctx.d_insertPrefixSizes));
        if (ctx.d_relocationPlan) checkCuda(cudaFree(ctx.d_relocationPlan));

        ctx.numRecords = newN;
        checkCuda(cudaMalloc(&ctx.d_nodes, newN * sizeof(CBSTNode)));
        checkCuda(cudaMalloc(&ctx.d_keys, newN * sizeof(int)));
        checkCuda(cudaMalloc(&ctx.d_startOffsets, newN * sizeof(int)));
        checkCuda(cudaMemcpy(ctx.d_keys, h_newKeys.data(), newN * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(ctx.d_startOffsets, h_newStarts.data(), newN * sizeof(int), cudaMemcpyHostToDevice));

        checkCuda(cudaMalloc(&ctx.d_avail, newN * sizeof(int)));
        checkCuda(cudaMalloc(&ctx.d_subtreeAvail, newN * sizeof(int)));
        checkCuda(cudaMemset(ctx.d_avail, 0, newN * sizeof(int)));
        checkCuda(cudaMemset(ctx.d_subtreeAvail, 0, newN * sizeof(int)));

        // Recreate insert buffers according to newN
        checkCuda(cudaMalloc(&ctx.d_insertKeys, newN * sizeof(int)));
        checkCuda(cudaMalloc(&ctx.d_insertPayload, newN * 3 * sizeof(int)));
        checkCuda(cudaMalloc(&ctx.d_insertPrefixSizes, newN * sizeof(int)));
        checkCuda(cudaMalloc(&ctx.d_relocationPlan, newN * 3 * sizeof(int)));

        int blocksBuild = (newN + blockSize - 1) / blockSize;
        buildEmptyBinaryTree<<<blocksBuild, blockSize>>>(ctx.d_nodes, newN);
        checkCuda(cudaDeviceSynchronize());
        storeItemsIntoNodes<<<blocksBuild, blockSize>>>(ctx.d_nodes, ctx.d_keys, ctx.d_startOffsets, newN, ctx.initialPayloadSize);
        checkCuda(cudaDeviceSynchronize());
    }
    // Recompute subtreeAvail bottom-up to reflect consumed deletions
    int blockNodes = (ctx.numRecords + blockSize - 1) / blockSize;
    reduceAvailLevel<<<blockNodes, blockSize>>>(ctx.numRecords - 1, ctx.numRecords - 1, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
    checkCuda(cudaDeviceSynchronize());
    int lastLevelStart = 1;
    while (lastLevelStart * 2 <= ctx.numRecords) lastLevelStart <<= 1;
    int levelStart = lastLevelStart - 1;
    if (levelStart >= ctx.numRecords) levelStart = (lastLevelStart >> 1) - 1;
    for (int levelEnd = ctx.numRecords - 1; levelStart >= 0; levelStart = (levelStart - 1) / 2) {
        int start = levelStart;
        int end = levelEnd;
        int count = end - start + 1;
        int blocks = (count + blockSize - 1) / blockSize;
        reduceAvailLevel<<<blocks, blockSize>>>(start, end, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
        checkCuda(cudaDeviceSynchronize());
        if (levelStart == 0) break;
        levelEnd = levelStart - 1;
        levelStart = (levelStart - 1) / 2;
    }
}

// CBSTOperations implementation
CBSTOperations::CBSTOperations(const char* datasetName, int payloadCapacity, int alignment) {
    ctx_.datasetName = datasetName;
    ctx_.fixedSize = payloadCapacity;
    ctx_.alignment = alignment;
}

CBSTOperations::~CBSTOperations() {
    if (ctx_.d_insertKeys)       checkCuda(cudaFree(ctx_.d_insertKeys));
    if (ctx_.d_insertPayload)    checkCuda(cudaFree(ctx_.d_insertPayload));
    if (ctx_.d_insertPrefixSizes)checkCuda(cudaFree(ctx_.d_insertPrefixSizes));
    if (ctx_.d_relocationPlan)   checkCuda(cudaFree(ctx_.d_relocationPlan));
    if (ctx_.d_keys)             checkCuda(cudaFree(ctx_.d_keys));
    if (ctx_.d_startOffsets)     checkCuda(cudaFree(ctx_.d_startOffsets));
    if (ctx_.d_nodes)            checkCuda(cudaFree(ctx_.d_nodes));
    if (ctx_.d_flatPayload)      checkCuda(cudaFree(ctx_.d_flatPayload));
}

CBSTOperations::CBSTOperations(CBSTOperations&& other) noexcept {
    ctx_ = other.ctx_;
    constructed_ = other.constructed_;
    // Null out other's pointers to avoid double free
    other.ctx_.d_nodes = nullptr;
    other.ctx_.d_keys = nullptr;
    other.ctx_.d_startOffsets = nullptr;
    other.ctx_.d_flatPayload = nullptr;
    other.ctx_.d_insertKeys = nullptr;
    other.ctx_.d_insertPayload = nullptr;
    other.ctx_.d_insertPrefixSizes = nullptr;
    other.ctx_.d_relocationPlan = nullptr;
    other.constructed_ = false;
}

CBSTOperations& CBSTOperations::operator=(CBSTOperations&& other) noexcept {
    if (this != &other) {
        // Free current resources
        this->~CBSTOperations();
        // Steal other's resources
        ctx_ = other.ctx_;
        constructed_ = other.constructed_;
        // Null out other's pointers
        other.ctx_.d_nodes = nullptr;
        other.ctx_.d_keys = nullptr;
        other.ctx_.d_startOffsets = nullptr;
        other.ctx_.d_flatPayload = nullptr;
        other.ctx_.d_insertKeys = nullptr;
        other.ctx_.d_insertPayload = nullptr;
        other.ctx_.d_insertPrefixSizes = nullptr;
        other.ctx_.d_relocationPlan = nullptr;
        other.constructed_ = false;
    }
    return *this;
}

void CBSTOperations::construct(int* keys, int* startOffsets, int numRecords, int* flatPayload, int flatPayloadSize) {
    constructCBST(keys, startOffsets, numRecords, flatPayload, flatPayloadSize, ctx_.fixedSize, ctx_.datasetName, ctx_);
    constructed_ = true;
}

void CBSTOperations::insert(const std::vector<int>& insertKeys, const std::vector<int>& insertPayload, const std::vector<int>& insertPrefixSizes) {
    fillCBST(insertKeys, insertPayload, insertPrefixSizes, ctx_);
}

void CBSTOperations::erase(const std::vector<int>& deleteKeys) {
    deleteCBST(deleteKeys, ctx_);
}

void CBSTOperations::findAndPrint(const std::vector<int>& ids) const {
    if (ids.empty()) return;
    int *d_search;
    checkCuda(cudaMalloc(&d_search, ids.size() * sizeof(int)));
    checkCuda(cudaMemcpy(d_search, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    findContents<<<(ids.size() + 256 - 1) / 256, 256>>>(ctx_.d_nodes, d_search, ids.size(), ctx_.d_flatPayload);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_search));
}

const CBSTContext& CBSTOperations::context() const {
    return ctx_;
}

void unfillCBST(const std::vector<int>& keysToUnfill,
                const std::vector<int>& valuesToRemove,
                const std::vector<int>& removePrefixSizes,
                CBSTContext& ctx) {
    if (keysToUnfill.empty()) return;
    // Reuse insert buffers for passing inputs
    checkCuda(cudaMemcpy(ctx.d_insertKeys, keysToUnfill.data(), keysToUnfill.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPayload, valuesToRemove.data(), valuesToRemove.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, removePrefixSizes.data(), removePrefixSizes.size() * sizeof(int), cudaMemcpyHostToDevice));
    int K = static_cast<int>(keysToUnfill.size());
    int blockSize = 256;
    int numBlocks = (K + blockSize - 1) / blockSize;
    unfillKernel<<<numBlocks, blockSize>>>(ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes, K);
    checkCuda(cudaDeviceSynchronize());
}


