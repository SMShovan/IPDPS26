# Figure 8: ESCHER vs MoCHy-E (Vary Delete Percentage)

This figure compares:

- `ESCHER` (update + delta counting)
- `MoCHy-E` (full recalculation)

across delete percentages `20, 40, 60, 80`.

## Files

- `figure8.cu`: benchmark driver that runs both ESCHER and MoCHy-E and writes timing data.
- `escher_vs_mochy_e_deletevary.txt`: canonical timing table.
- `figure8_compare.py`: reads txt and generates one plot per dataset.

## Output schema

Header:

`Mode Dataset DeletePercentage TimeSeconds`

Example row:

`ESCHER Coauth-DBLP 40 13.229`

## Generate timing table

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include -I../../kernel figure8.cu -o figure8
./figure8
```

`figure8.cu` compiles/runs MoCHy-E executable at:

`../../../MoCHy-master/exact_cli_par`

It auto-compiles this binary from `../../../MoCHy-master/main_exact_par.cpp` if missing.

## Generate plots

```bash
python3 figure8_compare.py
```
