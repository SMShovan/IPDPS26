#pragma once

// Common device/host helpers used by kernels

static inline __host__ __device__ int nextMultipleOf32(int num) {
    return ((num + 32) / 32) * 32;
}

static inline __host__ __device__ int nextMultipleOf4(int num) {
    if (num == 0) return 0;
    return ((num + 4) / 4) * 4;
}

static inline __device__ int ceil_log2(int x) {
    int log = 0;
    while ((1 << log) < x) ++log;
    return log;
}

static inline __device__ int floor_log2(int x) {
    int log = 0;
    while (x >>= 1) ++log;
    return log;
}


