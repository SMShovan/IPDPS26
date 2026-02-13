# Figure 15: baselineCUDA vs dynamicCUDA Speedup

This figure compares:

- `dynamicCUDA`: CHAI dynamic path (`construct + update + computeTemporalMotifCountsStrictIncDelta`)
- `baselineCUDA`: CUDA THyMe baseline from `THyMe-main/code/main_thymeP_cuda.cu`

across changed-edge sizes `50K`, `100K`, `200K`.

## Files

- `figure15.cu`: benchmark driver (`.cu -> .txt`)
- `cuda_thyme_speedup_timing.txt`: timing table consumed by plotting
- `figure15_compare.py`: computes speedup and renders grouped bars

## Timing schema

Header:

`Mode Dataset ChangedEdges TimeSeconds`

## Build and run

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include \
  figure15.cu \
  ../../src/temporal_count.cu \
  ../../src/temporal_structure.cpp \
  ../../utils/utils.cpp \
  ../../src/graphGeneration.cpp \
  ../../structure/operations.cu \
  -o figure15

./figure15
python3 figure15_compare.py
```
