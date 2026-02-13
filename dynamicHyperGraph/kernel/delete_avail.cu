#include "kernels.cuh"

__global__ void deleteNode(
    CBSTNode* nodes,
    int* deleteIndices,
    int deleteSize
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < deleteSize) {
        int deleteIndex = deleteIndices[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != deleteIndex) {
            if (current->index > deleteIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            current->index = -1;
            current = current->parent;
        }
    }
}

__global__ void markAvail(CBSTNode* nodes, int* deleteKeys, int deleteSize, int* avail) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < deleteSize) {
        int key = deleteKeys[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != key) {
            if (current->index > key) current = current->left; else current = current->right;
        }
        if (current != nullptr) {
            int idx = static_cast<int>(current - nodes);
            avail[idx] = 1;
        }
    }
}

__global__ void reduceAvailLevel(int levelStart, int levelEnd, int numRecords, int* avail, int* subtreeAvail) {
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = levelStart + g;
    if (idx < 0 || idx > levelEnd || idx >= numRecords) return;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    int leftSum = (left < numRecords) ? subtreeAvail[left] : 0;
    int rightSum = (right < numRecords) ? subtreeAvail[right] : 0;
    subtreeAvail[idx] = avail[idx] + leftSum + rightSum;
}


