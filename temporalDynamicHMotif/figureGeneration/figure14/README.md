# Figure 14: THyMe+ vs ESCHER Speedup (Changed Edges)

This figure compares:

- `THyME+` (OpenMP baseline, 8 threads)
- `ESCHER` (`temporalDynamicHMotif`, temporal construct + update + delta)

across changed-edge sizes `50K`, `100K`, `200K`.

## Files

- `figure14.cu`: benchmark driver (`.cu -> .txt`)
- `escher_vs_thymep_speedup_timing.txt`: timing table consumed by plotting
- `figure14_compare.py`: computes speedup and renders grouped bars

## Timing schema

Header:

`Mode Dataset ChangedEdges TimeSeconds`

## Build and run

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include \
  figure14.cu \
  ../../src/temporal_count.cu \
  ../../src/temporal_structure.cpp \
  ../../utils/utils.cpp \
  ../../src/graphGeneration.cpp \
  ../../structure/operations.cu \
  -o figure14

./figure14
python3 figure14_compare.py
```
