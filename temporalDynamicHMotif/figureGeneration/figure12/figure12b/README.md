# Figure 12b: Temporal Step-Wise Percentage Breakdown

This benchmark reports a per-dataset stacked percentage split across:

- `Construction`: temporal structure construction
- `Deletion`: deletion part of data-structure update
- `Insertion`: insertion part of data-structure update
- `Update`: `computeTemporalMotifCountsStrictIncDelta` runtime

The benchmark reads datasets from:

`../../../../dynamicHyperGraph/input/`

## Files

- `figure12b.cu`: benchmark driver (`.cu -> outputStepPercentage.txt`)
- `outputStepPercentage.txt`: plot input
- `figure12b.py`: reads txt and generates `stacked_percentage.pdf`

## Build and run

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../../include \
  figure12b.cu \
  ../../../src/temporal_count.cu \
  ../../../src/temporal_structure.cpp \
  ../../../utils/utils.cpp \
  ../../../src/graphGeneration.cpp \
  ../../../structure/operations.cu \
  -o figure12b

./figure12b
python3 figure12b.py
```
