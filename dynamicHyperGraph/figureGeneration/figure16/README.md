# Figure 16: Hornet vs ESCHER (Time + Memory Ratios)

This figure compares `Hornet` and `ESCHER` on random workload settings using:

- runtime (`TimeSeconds`)
- live-used memory (`MemoryMB`)

across `STD` values `[10, 20, 40, 60, 80]`.

## Files

- `figure16.cu`: benchmark driver (`.cu -> .txt`)
- `hornet_vs_escher_time_memory.txt`: table consumed by plotting script
- `figure16_compare.py`: computes time/memory ratios and renders line plot

## Timing and memory schema

Header:

`Mode Dataset STD TimeSeconds MemoryMB`

## Build and run

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include -I../../kernel \
  figure16.cu \
  ../../src/HMotifCount.cu \
  ../../src/HMotifCountUpdate.cu \
  ../../utils/utils.cpp \
  ../../utils/flatten.cpp \
  ../../utils/printUtils.cpp \
  ../../src/graphGeneration.cpp \
  ../../structure/operations.cu \
  -o figure16

./figure16
python3 figure16_compare.py
```
