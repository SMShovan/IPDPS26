# Top-level wrapper Makefile for IPDPS workspace

.PHONY: help all build-all clean-all \
	dynamic-build dynamic-run dynamic-figures \
	temporal-build temporal-run temporal-figures \
	temporal-figure12 temporal-figure13 temporal-figure14 temporal-figure15

help:
	@echo "Top-level targets:"
	@echo "  build-all        - Build dynamicHyperGraph and temporalDynamicHMotif"
	@echo "  clean-all        - Clean both subprojects"
	@echo "  dynamic-build    - Build dynamicHyperGraph"
	@echo "  dynamic-run      - Run dynamicHyperGraph default executable"
	@echo "  dynamic-figures  - Generate all configured dynamicHyperGraph figures"
	@echo "  temporal-build   - Build temporalDynamicHMotif"
	@echo "  temporal-run     - Run temporalDynamicHMotif default executable"
	@echo "  temporal-figures - Generate temporal figures (12a, 12b, 13, 14, 15)"
	@echo ""
	@echo "Examples:"
	@echo "  make dynamic-figures"
	@echo "  make temporal-figures"
	@echo "  make build-all"

all: build-all

build-all: dynamic-build temporal-build

clean-all:
	$(MAKE) -C dynamicHyperGraph clean
	$(MAKE) -C temporalDynamicHMotif clean

dynamic-build:
	$(MAKE) -C dynamicHyperGraph all

dynamic-run:
	$(MAKE) -C dynamicHyperGraph run

dynamic-figures:
	$(MAKE) -C dynamicHyperGraph figures-all

temporal-build:
	$(MAKE) -C temporalDynamicHMotif all

temporal-run:
	$(MAKE) -C temporalDynamicHMotif run

temporal-figures: temporal-figure12 temporal-figure13 temporal-figure14 temporal-figure15

temporal-figure12:
	cd temporalDynamicHMotif/figureGeneration/figure12/figure12a && \
	nvcc -std=c++17 -O2 -I../../../include \
	  figure12a.cu \
	  ../../../src/temporal_count.cu \
	  ../../../src/temporal_structure.cpp \
	  ../../../utils/utils.cpp \
	  ../../../utils/flatten.cpp \
	  ../../../utils/printUtils.cpp \
	  ../../../src/graphGeneration.cpp \
	  ../../../structure/operations.cu \
	  -o figure12a && \
	./figure12a && \
	python3 figure12a.py
	cd temporalDynamicHMotif/figureGeneration/figure12/figure12b && \
	nvcc -std=c++17 -O2 -I../../../include \
	  figure12b.cu \
	  ../../../src/temporal_count.cu \
	  ../../../src/temporal_structure.cpp \
	  ../../../utils/utils.cpp \
	  ../../../src/graphGeneration.cpp \
	  ../../../structure/operations.cu \
	  -o figure12b && \
	./figure12b && \
	python3 figure12b.py

temporal-figure13:
	cd temporalDynamicHMotif/figureGeneration/figure13 && \
	nvcc -std=c++17 -O2 -I../../include \
	  figure13.cu \
	  ../../src/temporal_count.cu \
	  ../../src/temporal_structure.cpp \
	  ../../utils/utils.cpp \
	  ../../src/graphGeneration.cpp \
	  ../../structure/operations.cu \
	  -o figure13 && \
	./figure13 && \
	python3 figure13_compare.py

temporal-figure14:
	cd temporalDynamicHMotif/figureGeneration/figure14 && \
	nvcc -std=c++17 -O2 -I../../include \
	  figure14.cu \
	  ../../src/temporal_count.cu \
	  ../../src/temporal_structure.cpp \
	  ../../utils/utils.cpp \
	  ../../src/graphGeneration.cpp \
	  ../../structure/operations.cu \
	  -o figure14 && \
	./figure14 && \
	python3 figure14_compare.py

temporal-figure15:
	cd temporalDynamicHMotif/figureGeneration/figure15 && \
	nvcc -std=c++17 -O2 -I../../include \
	  figure15.cu \
	  ../../src/temporal_count.cu \
	  ../../src/temporal_structure.cpp \
	  ../../utils/utils.cpp \
	  ../../src/graphGeneration.cpp \
	  ../../structure/operations.cu \
	  -o figure15 && \
	./figure15 && \
	python3 figure15_compare.py
