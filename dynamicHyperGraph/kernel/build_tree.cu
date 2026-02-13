#include "kernels.cuh"
#include "device_utils.cuh"
#include <cstdio>

__global__ void buildEmptyBinaryTree(CBSTNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        nodes[tid].index = tid;
        nodes[tid].left = (2 * tid + 1 < n) ? &nodes[2 * tid + 1] : nullptr;
        nodes[tid].right = (2 * tid + 2 < n) ? &nodes[2 * tid + 2] : nullptr;
        nodes[tid].parent = (tid == 0) ? nullptr : &nodes[(tid - 1) / 2];
    }
}

__global__ void storeItemsIntoNodes(CBSTNode* nodes, int* indices, int* values, int n, int totalSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int log2_tid = floor_log2(tid + 1);
        int log2_n = floor_log2(n);
        int index =  ((2 * (tid + 1  - (1<<log2_tid))) + 1) * (1 << log2_n) / (1 << log2_tid);
        int index2 = min(index, index - (index/2) + (n + 1 - (1<< log2_n)));
        index2--;
        nodes[tid].size = totalSize;
        if (index2 < n) {
            nodes[tid].index = indices[index2];
            nodes[tid].value = values[index2];
            if (index2 < n - 1) {
                nodes[tid].length = values[index2 + 1] - values[index2];
            } else {
                nodes[tid].length = totalSize - values[index2];
            }
        }
    }
}

__global__ void printEachNode(CBSTNode* nodes, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= n) {
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != tid) {
            if (current->index > tid) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            printf("Node %d: Index = %d, Value = %d, Length = %d, Size = %d\n",
                   tid, current->index, current->value, current->length, current->size);
        }
    }
}


