# Figure 7: ESCHER vs MoCHy-E (Vary Changed Edges)

This figure compares:

- `ESCHER` (update + delta counting)
- `MoCHy-E` (full recalculation)

across changed-edge sizes `50K`, `100K`, and `200K`.

## Files

- `figure7.cu`: benchmark driver that runs ESCHER and MoCHy-E and writes timing output.
- `escher_vs_mochy_e_timing.txt`: canonical timing table consumed by plotting code.
- `figure7_compare.py`: reads the txt and generates one PDF per dataset.

## Output schema

Header:

`Mode Dataset ChangedEdges TimeSeconds`

Example row:

`ESCHER Coauth-DBLP 50K 9.203`

## Generate timing table

From this directory:

```bash
nvcc -std=c++17 -O2 -I../../include -I../../kernel figure7.cu -o figure7
./figure7
```

`figure7.cu` compiles/runs MoCHy-E executable at:

`../../../MoCHy-master/exact_cli_par`

It auto-compiles this binary from `../../../MoCHy-master/main_exact_par.cpp` if missing.

## Generate plots

```bash
python3 figure7_compare.py
```
# Figure 12: ESCHER vs MoCHy-E

This folder contains the comparison plotting pipeline for:

- `ESCHER` (update-based method)
- `MoCHy-E` (recalculation-based baseline)

using changed-edge sizes `50K`, `100K`, and `200K`.

## Files

- `escher_vs_mochy_e_timing.txt`: timing table read by plotting script.
- `figure7_compare.py`: generates one PDF per dataset.

## Timing file format

Header:

`Mode Dataset ChangedEdges TimeSeconds`

Rows:

`<Mode> <Dataset> <ChangedEdges> <TimeSeconds>`

Example:

`ESCHER Coauth-DBLP 50K 9.203`

## Plot generation

From this directory:

```bash
python3 figure7_compare.py
```

## MoCHy input conversion utility

Use `utils/mochy_converter.cpp` to convert ESCHER space-separated hyperedge files to MoCHy comma-separated format:

```bash
g++ -O2 -std=c++17 utils/mochy_converter.cpp -o build/mochy_converter
./build/mochy_converter input/Coauth.txt MoCHy-Master/data/Coauth.csv
```
