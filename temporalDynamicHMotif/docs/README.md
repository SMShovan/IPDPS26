# Temporal Dynamic Hypergraph Motif Analysis – Developer Guide

## Overview

This project performs GPU-accelerated motif counting on dynamic hypergraphs and extends to temporal motifs across three timestamps (Older, Middle, Newest) with strictly-increasing time ordering. Data is stored in GPU-resident Complete Binary Search Trees (CBST) for efficient updates.

Key components:
- Hypergraph maps: H2V (hyperedge→vertices), V2H (vertex→hyperedges), H2H (adjacency for baseline)
- Temporal ring buffer: distinct H2V/V2H per layer, rotate each step
- Temporal motifs: 30-bin counts for strictly-increasing triples (t-2, t-1, t)

## Build

```bash
make clean && make -j
```

## Run

Baseline (non-temporal):
```bash
make run
make run-custom ARGS="<N> <K> <vmin> <vmax> <capacity> [alignment=4]"
```

Temporal:
```bash
make run-temporal
make run-temporal-deltas DELTA=updates.txt
make run-temporal-deltas-synth DELTA=updates_synth.txt
```

Direct execution (temporal flags):
```bash
./build/main <N> <K> <vmin> <vmax> <capacity> [alignment=4] --temporal [--temporal-synthetic] [--temporal-deltas=PATH]
```

Delta file format:
```
A <hyperedgeId> v1 v2 v3 ...  # add vertices to H2V[Newest][h]
R <hyperedgeId> v1 v2 v3 ...  # remove vertices from H2V[Newest][h]
```

## Code Map

- include/
  - structure.hpp: CBST structures and host APIs
  - temporal_structure.hpp: 3-slot ring buffer wrappers for H2V/V2H
  - temporal_adjacency.hpp: temporal index bundle (H2V/V2H)
  - temporal_count.hpp: temporal counting interfaces
  - motif.hpp, motif_update.hpp: baseline motif counting
  - utils.hpp, printUtils.hpp, graphGeneration.hpp
- structure/operations.cu: CBST kernels (construct/fill/unfill)
- kernel/: device utilities and baseline kernels
- src/
  - main.cu: entry point (baseline + temporal flows)
  - temporal_count.cu: adjacency-driven temporal kernel + delta ingestion
  - temporal_structure.cpp: host wrappers for temporal CBSTs
  - HMotifCount.cu, HMotifCountUpdate.cu, graphGeneration.cpp
- utils/: flatten/print/utils

## Notes

- H2H is computed on-the-fly for baseline. Temporal path uses H2V/V2H intersections on device.
- Deltas are mirrored to V2H for the Newest layer to drive adjacency on GPU.
