# Project Structure

This document describes the organization of the Dynamic Hypergraph Analysis project.

## Directory Layout

```
dynamicHyperGraph/
├── include/                    # Header files (.hpp)
│   ├── graphGeneration.hpp    # Graph generation utilities
│   ├── printUtils.hpp         # Printing utilities
│   └── utils.hpp              # General utilities and data structures
├── src/                       # Main source files
│   ├── graphGeneration.cpp    # Graph generation implementation
│   └── main.cu               # Main CUDA program
├── utils/                     # Utility implementations
│   ├── printUtils.cpp        # Printing functions
│   └── utils.cpp             # General utility functions
├── build/                     # Build artifacts (generated)
├── Makefile                   # Build configuration
└── README.md                  # Project documentation
```

## File Organization

### Headers (`include/`)
- **`graphGeneration.hpp`**: Declarations for hypergraph generation functions
- **`printUtils.hpp`**: Declarations for printing and display functions
- **`utils.hpp`**: Declarations for general utilities, data structures, and main workflow functions

### Source Files (`src/`)
- **`main.cu`**: Main CUDA program with kernels and GPU processing logic
- **`graphGeneration.cpp`**: Implementation of hypergraph generation algorithms

### Utilities (`utils/`)
- **`utils.cpp`**: Implementation of command-line parsing, hypergraph generation workflow, and CBST data preparation
- **`printUtils.cpp`**: Implementation of printing and display functions

## Benefits of This Structure

1. **Separation of Concerns**: Headers, main logic, and utilities are clearly separated
2. **Modularity**: Each component can be developed and tested independently
3. **Maintainability**: Easy to locate and modify specific functionality
4. **Scalability**: Easy to add new utilities or modules
5. **Professional**: Follows standard C++ project organization practices

## Build Process

The Makefile automatically:
- Compiles CUDA files with `-Iinclude` flag for header resolution
- Compiles C++ files with proper include paths
- Links all object files into the final executable
- Handles dependencies between files
