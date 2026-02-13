#pragma once

#include "../include/structure.hpp"

// Build tree
__global__ void buildEmptyBinaryTree(CBSTNode* nodes, int n);
__global__ void storeItemsIntoNodes(CBSTNode* nodes, int* indices, int* values, int n, int totalSize);
__global__ void printEachNode(CBSTNode* nodes, int n);

// Payload ops
__global__ void insertNode(CBSTNode* nodes, int* flatValues, int* insertIndices, int* insertValues, int* insertSizes, int insertSize, int* partialSolution);
__global__ void allocateSpace(int* partialSolution, int* flatValues, int spaceAvailableFrom, int* insertIndices, int* insertValues, int* insertSizes, int insertSize);
__global__ void computeNextMultipleOf4(int* partialSolution, int* tmp, int K);
__global__ void updatePartialSolution(int* partialSolution, int* tmp, int K);

// Delete / availability
__global__ void deleteNode(CBSTNode* nodes, int* deleteIndices, int deleteSize);
__global__ void markAvail(CBSTNode* nodes, int* deleteKeys, int deleteSize, int* avail);
__global__ void reduceAvailLevel(int levelStart, int levelEnd, int numRecords, int* avail, int* subtreeAvail);

// Lookup / find
__global__ void findNode(CBSTNode* nodes, int* searchIndices, int searchSize);
__global__ void findContents(CBSTNode* nodes, int* searchIndices, int searchSize, int* flatValues);

// Insert reuse and unfill
__global__ void insertIntoDeletedKth(CBSTNode* nodes,
                                     int* flatValues,
                                     int* subtreeAvail,
                                     int* avail,
                                     int numRecords,
                                     int* newKeys,
                                     int* newPayload,
                                     int* newPrefixSizes,
                                     int* relocationPlan,
                                     int K);

__global__ void unfillKernel(CBSTNode* nodes,
                             int* flatValues,
                             int* keys,
                             int* valuesToRemove,
                             int* removePrefixSizes,
                             int K);


