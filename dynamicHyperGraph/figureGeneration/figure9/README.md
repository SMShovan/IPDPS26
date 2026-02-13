# Figure 9: Speedup (Baseline vs ESCHER)

This folder contains a txt-driven speedup plotting pipeline.

## Files

- `figure9.cu`: emits the canonical timing table for this figure.
- `speedup_timing.txt`: timing data table consumed by plotting script.
- `figure9_compare.py`: reads the txt, computes speedup, and generates `Speedup.pdf`.

## Timing schema

Header:

`Mode Dataset ChangedEdges TimeSeconds`

Example:

`ESCHER Coauth 50K 9.203`

## Generate timing file

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include -I../../kernel figure9.cu -o figure9
./figure9
```

`figure9.cu` compiles/runs baseline executable at:

`../../../MoCHy-master/exact_cli_par`

It auto-compiles this binary from `../../../MoCHy-master/main_exact_par.cpp` if missing.

## Generate plot

```bash
python3 figure9_compare.py
```
