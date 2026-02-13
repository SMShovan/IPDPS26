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
               $(KERNEL_DIR)/delete_avail.cu $(KERNEL_DIR)/find.cu
CPP_SOURCES = $(SRC_DIR)/graphGeneration.cpp $(UTILS_DIR)/utils.cpp $(UTILS_DIR)/printUtils.cpp
HEADERS = $(INCLUDE_DIR)/graphGeneration.hpp $(INCLUDE_DIR)/utils.hpp $(INCLUDE_DIR)/printUtils.hpp

# Object files
CUDA_OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/HMotifCount.o $(BUILD_DIR)/HMotifCountUpdate.o $(BUILD_DIR)/operations.o \
               $(BUILD_DIR)/insert_reuse.o $(BUILD_DIR)/unfill.o \
               $(BUILD_DIR)/payload.o $(BUILD_DIR)/build_tree.o \
               $(BUILD_DIR)/delete_avail.o $(BUILD_DIR)/find.o
CPP_OBJECTS = $(BUILD_DIR)/graphGeneration.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/printUtils.o

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

# --------------------------
# Figure generation targets
# --------------------------

figures-all: figure6 figure7 figure8 figure9 figure10 figure11 figure16

figure6: figure6a figure6b figure6c figure6d

figure6a:
	cd figureGeneration/figure6/figure6a && \
	nvcc -std=c++17 -O2 -I../../../include -I../../../kernel figure6a.cu -o figure6a && \
	./figure6a && \
	python3 figure6a.py

figure6b:
	cd figureGeneration/figure6/figure6b && \
	nvcc -std=c++17 -O2 -I../../../include -I../../../kernel figure6b.cu -o figure6b && \
	./figure6b && \
	python3 figure6b.py

figure6c:
	cd figureGeneration/figure6/figure6c && \
	nvcc -std=c++17 -O2 -I../../../include -I../../../kernel figure6c.cu -o figure6c && \
	./figure6c && \
	python3 figure6c.py

figure6d:
	cd figureGeneration/figure6/figure6d && \
	nvcc -std=c++17 -O2 -I../../../include -I../../../kernel figure6d.cu -o figure6d && \
	./figure6d && \
	python3 figure6d.py

figure7:
	cd figureGeneration/figure7 && \
	nvcc -std=c++17 -O2 -I../../include -I../../kernel figure7.cu -o figure7 && \
	./figure7 && \
	python3 figure7_compare.py

figure8:
	cd figureGeneration/figure8 && \
	nvcc -std=c++17 -O2 -I../../include -I../../kernel figure8.cu -o figure8 && \
	./figure8 && \
	python3 figure8_compare.py

figure9:
	cd figureGeneration/figure9 && \
	nvcc -std=c++17 -O2 -I../../include -I../../kernel figure9.cu -o figure9 && \
	./figure9 && \
	python3 figure9_compare.py

figure10:
	cd figureGeneration/figure10 && \
	nvcc -std=c++17 -O2 -I../../include -I../../kernel figure10.cu -o figure10 && \
	./figure10 && \
	python3 figure10_compare.py

figure11:
	cd figureGeneration/figure11 && \
	nvcc -std=c++17 -O2 figure11.cu -o figure11 && \
	./figure11 && \
	python3 figure11_compare.py

figure16:
	cd figureGeneration/figure16 && \
	nvcc -std=c++17 -O2 -I../../include -I../../kernel \
	  figure16.cu \
	  ../../src/HMotifCount.cu \
	  ../../src/HMotifCountUpdate.cu \
	  ../../utils/utils.cpp \
	  ../../utils/flatten.cpp \
	  ../../utils/printUtils.cpp \
	  ../../src/graphGeneration.cpp \
	  ../../structure/operations.cu \
	  -o figure16 && \
	./figure16 && \
	python3 figure16_compare.py

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
	@echo "  figures-all - Build/run all figure pipelines (6,7,8,9,10,11,16)"
	@echo "  figure6     - Build/run all subfigures (6a, 6b, 6c, 6d)"
	@echo "  figure7     - Build/run figure 7 benchmark + plot"
	@echo "  figure8     - Build/run figure 8 benchmark + plot"
	@echo "  figure9     - Build/run figure 9 benchmark + plot"
	@echo "  figure10    - Build/run figure 10 benchmark + plot"
	@echo "  figure11    - Build/run figure 11 benchmark + plot"
	@echo "  figure16    - Build/run figure 16 benchmark + plot"
	@echo "  clean       - Remove build artifacts"
	@echo "  rebuild     - Clean and rebuild"
	@echo "  install     - Show installation instructions"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make run                    # Run with default: 8 hyperedges, 5 max vertices, IDs 1-100, capacity 4096"
	@echo "  make run-custom ARGS=\"10 3 1 50 8192\"  # Run with: 10 hyperedges, 3 max vertices, IDs 1-50, capacity 8192"
	@echo "  make figures-all            # Generate all figures in one command"
	@echo "  make figure10               # Generate figure10 only"
	@echo "  ./build/main 20 4 1 200 16384    # Direct execution with custom capacity"

# Phony targets
.PHONY: all run run-custom figures-all figure6 figure6a figure6b figure6c figure6d figure7 figure8 figure9 figure10 figure11 figure16 clean rebuild install help
