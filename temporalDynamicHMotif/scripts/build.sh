#!/bin/bash

echo "=== Dynamic Hypergraph Analysis Project Build Test ==="
echo

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found. Please ensure CUDA toolkit is installed."
    echo "   On cluster, run: module load cuda-toolkit/12.5"
    exit 1
fi

echo "✅ nvcc found: $(which nvcc)"

# Clean previous build
echo "🧹 Cleaning previous build..."
make clean

# Build the project
echo "🔨 Building project..."
if make; then
    echo "✅ Build successful!"
    echo
    echo "📁 Project structure:"
    echo "   ├── src/                   (Source code)"
    echo "   │   ├── main.cu           (Main CUDA implementation)"
    echo "   │   ├── graphGeneration.hpp (Header file)"
    echo "   │   └── graphGeneration.cpp (C++ implementation)"
    echo "   ├── build/                (Build artifacts)"
    echo "   │   └── main              (Executable)"
    echo "   ├── scripts/              (Build scripts)"
    echo "   ├── docs/                 (Documentation)"
    echo "   └── Makefile             (Build configuration)"
    echo
    echo "🚀 To run the program:"
    echo "   make run                    # Default: 8 hyperedges, 5 max vertices, IDs 1-100"
    echo "   make run-custom ARGS=\"10 3 1 50\"  # Custom parameters"
    echo "   ./build/main 20 4 1 200    # Direct execution"
    echo
    echo "📋 Available make targets:"
    echo "   make all         - Build the project"
    echo "   make run         - Build and run with default parameters"
    echo "   make run-custom  - Run with custom parameters"
    echo "   make clean       - Clean build artifacts"
    echo "   make rebuild     - Clean and rebuild"
    echo "   make help        - Show all targets"
else
    echo "❌ Build failed!"
    exit 1
fi
