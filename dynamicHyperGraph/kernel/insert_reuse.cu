#include "kernels.cuh"
#include <climits>

__global__ void insertIntoDeletedKth(CBSTNode* nodes,
                                     int* flatValues,
                                     int* subtreeAvail,
                                     int* avail,
                                     int numRecords,
                                     int* newKeys,
                                     int* newPayload,
                                     int* newPrefixSizes,
                                     int* relocationPlan,
                                     int K) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= K) return;
    int k = tid + 1; // 1-based order statistic
    int idx = 0;
    while (idx < numRecords) {
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;
        int leftCount = (left < numRecords) ? subtreeAvail[left] : 0;
        int self = avail[idx];
        if (k <= leftCount) {
            idx = left;
            continue;
        }
        if (self == 1 && k == leftCount + 1) {
            break; // found deleted node at idx
        }
        k -= leftCount + self;
        idx = right;
    }
    CBSTNode* node = &nodes[idx];
    int key = newKeys[tid];
    int start = (tid == 0) ? 0 : newPrefixSizes[tid - 1];
    int end = newPrefixSizes[tid];
    int len = end - start;
    int base = node->value;
    int capacity = node->length - 1; // leave space for INT_MIN
    if (len <= capacity) {
        for (int i = 0; i < len; ++i) {
            flatValues[base + i] = newPayload[start + i];
        }
        flatValues[base + len] = INT_MIN;
    } else {
        for (int i = 0; i < capacity; ++i) {
            flatValues[base + i] = newPayload[start + i];
        }
        int idx3 = tid * 3;
        relocationPlan[idx3] = base + capacity;        // location to place negative back-pointer
        relocationPlan[idx3 + 1] = capacity;           // start offset in this payload
        relocationPlan[idx3 + 2] = len - capacity;     // remaining length to append
    }
    node->index = key;
    avail[idx] = 0;
}


