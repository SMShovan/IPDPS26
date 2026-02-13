#include "kernels.cuh"
#include <climits>

__global__ void unfillKernel(CBSTNode* nodes,
                             int* flatValues,
                             int* keys,
                             int* valuesToRemove,
                             int* removePrefixSizes,
                             int K) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= K) return;
    int key = keys[tid];
    CBSTNode* cur = nodes;
    while (cur != nullptr && cur->index != key) {
        cur = (cur->index > key) ? cur->left : cur->right;
    }
    if (cur == nullptr) return;
    int segBase = cur->value;
    int start = (tid == 0) ? 0 : removePrefixSizes[tid - 1];
    int end = removePrefixSizes[tid];
    for (;;) {
        int w = 0;
        int i = 0;
        for (;;) {
            int val = flatValues[segBase + i];
            if (val == INT_MIN || val == 0 || val < 0) break;
            bool removeIt = false;
            for (int r = start; r < end; ++r) {
                if (valuesToRemove[r] == val) { removeIt = true; break; }
            }
            if (!removeIt) {
                flatValues[segBase + w] = val;
                ++w;
            }
            ++i;
        }
        int endVal = flatValues[segBase + i];
        if (endVal == INT_MIN) {
            flatValues[segBase + w] = INT_MIN;
            for (int z = w + 1; z < i; ++z) flatValues[segBase + z] = 0;
            break;
        } else if (endVal < 0) {
            for (int z = w; z < i; ++z) flatValues[segBase + z] = 0;
            segBase = -endVal;
            continue;
        } else {
            flatValues[segBase + w] = INT_MIN;
            for (int z = w + 1; z < i; ++z) flatValues[segBase + z] = 0;
            break;
        }
    }
}


