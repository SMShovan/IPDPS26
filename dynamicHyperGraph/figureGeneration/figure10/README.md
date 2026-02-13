# Figure 10: GPU Update vs GPU Recalculation

This figure compares two GPU variants on the same ESCHER codebase:

- `dynamicCUDA`: update + delta counting
- `baselineCUDA`: full recalculation on changed hypergraph

## Files

- `figure10.cu`: benchmark driver that writes timing data.
- `gpu_vs_gpu_timing.txt`: canonical timing table consumed by plotting script.
- `figure10_compare.py`: reads txt, computes speedup, and generates `SpeedupCUDA.pdf`.

## Timing schema

Header:

`Mode Dataset ChangedEdges TimeSeconds`

Example row:

`dynamicCUDA Coauth 50K 9.203`

## Generate timing file

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include -I../../kernel figure10.cu -o figure10
./figure10
```

## Generate plot

```bash
python3 figure10_compare.py
```
