# Figure 11: Type-wise Speedup Breakdown

This figure reports speedup per motif type family:

- Type 1
- Type 2
- Type 3

across datasets and changed-edge sizes.

## Files

- `figure11.cu`: benchmarks type-wise dynamic-vs-baseline paths and writes speedup table.
- `figure11_types_speedup.txt`: table consumed by plotting code.
- `figure11_compare.py`: reads txt and generates nested grouped bar plot.

## Timing schema

Header:

`Dataset ChangedEdges Type1Speedup Type2Speedup Type3Speedup`

Example row:

`Coauth 50K 182.03 282.03 312.03`

## Generate timing file

From this directory:

```bash
nvcc -std=c++17 -O2 figure11.cu -o figure11
./figure11
```

## Generate plot

```bash
python3 figure11_compare.py
```
