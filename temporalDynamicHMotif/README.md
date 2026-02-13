# Temporal Dynamic Hypergraph Motif Analysis (CUDA)

GPU-accelerated motif counting on dynamic hypergraphs using Complete Binary Search Trees (CBST) and CUDA. Supports temporal (3-timestamp) strictly-increasing motif counting with ring-buffer rotation and delta ingestion.

## Requirements

- CUDA Toolkit 12.x (tested on 12.5–12.8)
- NVIDIA GPU with CUDA support
- g++ (for host C++ files)

## Build

```bash
make clean && make -j
```

Artifacts are placed in `build/` (e.g., `build/main`).

## Run (baseline)

```bash
# Make target (defaults):
make run

# Custom args via make:
make run-custom ARGS="<num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4]"

# Direct execution:
./build/main <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4]

# Example
./build/main 20 5 1 200 16384 8
```

## Run (temporal)

The temporal path computes 30-bin counts for strictly-increasing timestamps across a 3-layer window (Older, Middle, Newest) and supports deltas applied to Newest.

```bash
# Enable temporal mode
make run-temporal

# Temporal with external deltas file
make run-temporal-deltas DELTA=updates.txt

# Synthetic temporal window (deterministic) + optional deltas (guaranteed signal)
make run-temporal-deltas-synth DELTA=updates_synth.txt

# Direct execution form
./build/main <N> <K> <vmin> <vmax> <capacity> [alignment=4] --temporal [--temporal-synthetic] [--temporal-deltas=PATH]
```

CLI flags:
- `--temporal`: enable temporal counting outputs
- `--temporal-synthetic`: use a small deterministic 3-edge window (useful for validation)
- `--temporal-deltas=PATH`: apply vertex add/remove deltas per hyperedge into Newest

Delta file format (space-separated lines):
```
A <hyperedgeId> v1 v2 v3 ...   # add vertices to H2V[Newest][hyperedgeId]
R <hyperedgeId> v1 v2 v3 ...   # remove vertices from H2V[Newest][hyperedgeId]
```

## Outputs

The program prints:
- Flattened arrays and CBST summaries for H2V, V2H, H2H
- Baseline motif counts (30 bins)
- If `--temporal`:
  - Temporal strict-inc counts (30 bins)
  - Temporal strict-inc delta (30 bins) between old/new windows

## Project Structure

```
temporalDynamicHMotif/
├── include/
│   ├── graphGeneration.hpp        # Host graph generation helpers
│   ├── motif.hpp, motif_update.hpp# Baseline motif counting/update APIs
│   ├── printUtils.hpp, utils.hpp  # Printing + CLI utilities
│   ├── structure.hpp              # CBST types and host interfaces
│   ├── temporal_structure.hpp     # 3-slot ring buffer wrappers (H2V, V2H)
│   ├── temporal_adjacency.hpp     # Temporal index container (H2V/V2H)
│   └── temporal_count.hpp         # Temporal counting interfaces
├── kernel/
│   ├── build_tree.cu, delete_avail.cu, find.cu
│   ├── insert_reuse.cu, payload.cu, unfill.cu
│   ├── device_utils.cuh, kernels.cuh, motif_utils.cuh
│   └── motifs.cu                  # Baseline motif kernels
├── structure/
│   └── operations.cu              # CBST construct/fill/unfill kernels
├── src/
│   ├── main.cu                    # Entry point (baseline + temporal)
│   ├── graphGeneration.cpp        # Host graph helpers
│   ├── HMotifCount.cu, HMotifCountUpdate.cu
│   ├── temporal_structure.cpp     # Temporal wrappers (host-side)
│   └── temporal_count.cu          # Temporal adjacency-driven kernel + I/O
├── utils/
│   ├── flatten.cpp, printUtils.cpp, utils.cpp
├── docs/                          # Extended documentation
├── scripts/                       # Helper scripts
├── build/                         # Build artifacts (generated)
└── Makefile                       # Build and run targets
```

## Notes

- H2H is computed on-the-fly for baseline; temporal counting uses H2V/V2H intersections on device.
- Temporal ring buffer maintains distinct H2V/V2H per layer; deltas apply to Newest and are mirrored to V2H.

## License

This project is part of academic research in hypergraph analysis.