# Figure 13: ESCHER vs THyME+ (Delete Percentage Vary)

This figure compares:

- `THyME+` (OpenMP baseline, 8 threads)
- `ESCHER` (`temporalDynamicHMotif`, temporal construct + update + delta)

across delete percentages `20, 40, 60, 80`.

## Files

- `figure13.cu`: benchmark driver (`.cu -> .txt`)
- `escher_vs_thymep_deletevary_temporal.txt`: timing table consumed by plotting
- `figure13_compare.py`: per-dataset line plots

## Timing schema

Header:

`Mode Dataset DeletePercentage TimeSeconds`

## Build and run

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include \
  figure13.cu \
  ../../src/temporal_count.cu \
  ../../src/temporal_structure.cpp \
  ../../utils/utils.cpp \
  ../../src/graphGeneration.cpp \
  ../../structure/operations.cu \
  -o figure13

./figure13
python3 figure13_compare.py
```
