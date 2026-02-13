# Makefile for Dynamic Hypergraph Analysis Project
# Compiler and flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++17 -O2
CXX_FLAGS = -std=c++17 -O2 -Wall

# Directories
SRC_DIR = src
UTILS_DIR = utils
INCLUDE_DIR = include
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main

# Source files
STRUCT_DIR = structure
KERNEL_DIR = kernel
CUDA_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/HMotifCount.cu $(SRC_DIR)/HMotifCountUpdate.cu $(STRUCT_DIR)/operations.cu \
               $(KERNEL_DIR)/insert_reuse.cu $(KERNEL_DIR)/unfill.cu \
               $(KERNEL_DIR)/payload.cu $(KERNEL_DIR)/build_tree.cu \
               $(KERNEL_DIR)/delete_avail.cu $(KERNEL_DIR)/find.cu \
               $(SRC_DIR)/temporal_count.cu
CPP_SOURCES = $(SRC_DIR)/graphGeneration.cpp $(UTILS_DIR)/utils.cpp $(UTILS_DIR)/printUtils.cpp $(SRC_DIR)/temporal_structure.cpp
HEADERS = $(INCLUDE_DIR)/graphGeneration.hpp $(INCLUDE_DIR)/utils.hpp $(INCLUDE_DIR)/printUtils.hpp

# Object files
CUDA_OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/HMotifCount.o $(BUILD_DIR)/HMotifCountUpdate.o $(BUILD_DIR)/operations.o \
               $(BUILD_DIR)/insert_reuse.o $(BUILD_DIR)/unfill.o \
               $(BUILD_DIR)/payload.o $(BUILD_DIR)/build_tree.o \
               $(BUILD_DIR)/delete_avail.o $(BUILD_DIR)/find.o \
               $(BUILD_DIR)/temporal_count.o
CPP_OBJECTS = $(BUILD_DIR)/graphGeneration.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/printUtils.o $(BUILD_DIR)/temporal_structure.o

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -Ikernel -c $< -o $@

# Compile CUDA source files under structure/
$(BUILD_DIR)/%.o: $(STRUCT_DIR)/%.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -Ikernel -c $< -o $@

# Compile CUDA source files under kernel/
$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -Ikernel -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile utility source files
$(BUILD_DIR)/%.o: $(UTILS_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Run the program with default parameters (includes payload capacity)
run: $(TARGET)
	$(TARGET) 8 5 1 100 4096

# Run with custom parameters (usage: make run-custom ARGS="10 3 1 50 8192")
run-custom: $(TARGET)
	$(TARGET) $(ARGS)

# Run with temporal counting enabled (sets a runtime flag)
run-temporal: $(TARGET)
	$(TARGET) 8 5 1 100 4096 --temporal

# Run temporal with deltas file passed as DELTA=path/to/updates.txt
run-temporal-deltas: $(TARGET)
	$(TARGET) 8 5 1 100 4096 --temporal --temporal-deltas=$(DELTA)

# Synthetic window + deltas for guaranteed signal (pass DELTA=updates_synth.txt)
run-temporal-deltas-synth: $(TARGET)
	$(TARGET) 8 5 1 100 4096 --temporal --temporal-synthetic --temporal-deltas=$(DELTA)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Clean and rebuild
rebuild: clean all

# Install dependencies (if needed)
install:
	@echo "Make sure CUDA toolkit is installed and nvcc is in PATH"

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the project (default)"
	@echo "  run         - Build and run with default parameters (8 5 1 100 4096)"
	@echo "  run-custom  - Run with custom parameters (make run-custom ARGS=\"10 3 1 50 8192\")"
	@echo "  clean       - Remove build artifacts"
	@echo "  rebuild     - Clean and rebuild"
	@echo "  install     - Show installation instructions"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make run                    # Run with default: 8 hyperedges, 5 max vertices, IDs 1-100, capacity 4096"
	@echo "  make run-custom ARGS=\"10 3 1 50 8192\"  # Run with: 10 hyperedges, 3 max vertices, IDs 1-50, capacity 8192"
	@echo "  ./build/main 20 4 1 200 16384    # Direct execution with custom capacity"

# Phony targets
.PHONY: all run run-custom clean rebuild install help
