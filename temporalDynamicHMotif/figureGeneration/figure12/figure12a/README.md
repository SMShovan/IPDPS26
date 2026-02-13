# Figure 12a: Temporal Strict-Inc Delta Timing

This benchmark measures total time as:

1. construct temporal windows,
2. apply 50/50 delete-insert update to build new window state,
3. run `computeTemporalMotifCountsStrictIncDelta`.

The benchmark reads datasets from the shared input folder:

`../../../../dynamicHyperGraph/input/`

## Files

- `figure12a.cu`: benchmark driver (`.cu -> output*.txt`)
- `outputCoauth.txt`
- `outputTags.txt`
- `outputOrkut.txt`
- `outputThreads.txt`
- `outputRandom.txt`
- `figure12a.py`: reads output files and generates `time_vs_DeltaE.pdf`

## Build and run

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../../include \
  figure12a.cu \
  ../../../src/temporal_count.cu \
  ../../../src/temporal_structure.cpp \
  ../../../utils/utils.cpp \
  ../../../utils/flatten.cpp \
  ../../../utils/printUtils.cpp \
  ../../../src/graphGeneration.cpp \
  ../../../structure/operations.cu \
  -o figure12a

./figure12a
python3 figure12a.py
```
