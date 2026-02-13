#include "kernels.cuh"
#include <climits>
#include <cstdio>

__global__ void findNode(CBSTNode* nodes, int* searchIndices, int searchSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < searchSize) {
        int searchIndex = searchIndices[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != searchIndex) {
            if (current->index > searchIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            printf("Node %d: Index = %d, Value = %d, Length = %d\n",
                   searchIndex, current->index, current->value, current->length);
        } else {
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}

__global__ void findContents(CBSTNode* nodes, int* searchIndices, int searchSize, int* flatValues) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < searchSize) {
        int searchIndex = searchIndices[tid];
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != searchIndex) {
            if (current->index > searchIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            int currLoc = current->value;
            printf("\n");
            while(flatValues[currLoc++] != INT_MIN)
            {
                printf("%d ", flatValues[currLoc]);
            }
            printf("\n");
            printf("Node %d: Index = %d, Value = %d, Length = %d\n", searchIndex, current->index, current->value, current->length);
        } else {
            printf("Node %d: Not Found\n", searchIndex);
        }
    }
}


