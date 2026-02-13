# Dynamic Hypergraph Analysis Project

A CUDA-based implementation for motif counting in dynamic hypergraphs using Complete Binary Search Trees and parallel processing.

## Quick Start

```bash
# Build the project
make

# Run the program with default parameters
make run

# Run with custom parameters (generate mode)
./build/main <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4] [--save-generated[=FILE]]
# Optional update-batch knobs:
#   -totalChanges N -insertionPercentage P -deletionPercentage P

# Example: 10 hyperedges, up to 3 vertices each, vertex IDs 1-50
./build/main 10 3 1 50 8192
./build/main 10 3 1 50 8192 -totalChanges 100 -insertionPercentage 70 -deletionPercentage 30

# Save generated graph under input/
./build/main 10 3 1 50 8192 --save-generated=sample.txt

# Input mode (infer graph parameters from input/sample.txt)
./build/main 8192 --input=sample.txt
# Optional alignment in input mode
./build/main 8192 8 --input=sample.txt
# Optional update-batch knobs in input mode
./build/main 8192 --input=sample.txt -totalChanges 50 -insertionPercentage 40 -deletionPercentage 60

# Clean build artifacts
make clean
```

## Project Structure

```
DynamicHypergraphMotif/
├── src/                    # Source code
│   ├── main.cu            # Main CUDA implementation
│   ├── graphGeneration.hpp # Graph generation header
│   └── graphGeneration.cpp # Graph generation implementation
├── build/                  # Build artifacts (auto-generated)
├── scripts/                # Build and utility scripts
├── docs/                   # Documentation
│   └── README.md          # Detailed documentation
├── Makefile               # Build configuration
└── .gitignore            # Git ignore file
```

## Documentation

For detailed documentation, see [docs/README.md](docs/README.md).

## Requirements

- CUDA Toolkit 12.5+
- NVIDIA GPU with CUDA support
- Thrust library (included with CUDA)

## License

This project is part of academic research in hypergraph analysis.