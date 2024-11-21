# Compiler selection
CXX = g++

# Use g++-12 on macOS if available
ifeq ($(shell uname), Darwin)
    CXX = g++-12
endif

# Fall back to g++ if specified compiler not found
ifeq (, $(shell which $(CXX)))
    CXX = g++
endif

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I include $(shell pkg-config --cflags armadillo) -fopenmp

# Linker flags
LDFLAGS = $(shell pkg-config --libs armadillo) -fopenmp

# Directory structure
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include
DATA_DIR = data
FIGURES_DIR = figures

# Source files
SRC_ISING = $(SRC_DIR)/IsingModel.cpp
SRC_ISING_TESTS = $(SRC_DIR)/IsingTests.cpp
SRC_BURNING = $(SRC_DIR)/Burning.cpp
SRC_ENERGY_DIST = $(SRC_DIR)/EnergyDist.cpp
SRC_PARALLEL = $(SRC_DIR)/Parallel.cpp
SRC_PHASE_TRANS = $(SRC_DIR)/PhaseTrans.cpp
SRC_PHASE_TRANS_IMPROVED = $(SRC_DIR)/PhaseTransImproved.cpp

# Target executables
TARGET_ISING = $(BUILD_DIR)/ising.exe
TARGET_BURNIN = $(BUILD_DIR)/burning.exe
TARGET_ENERGY_DIST = $(BUILD_DIR)/energy_dist.exe
TARGET_PARALLEL = $(BUILD_DIR)/parallel.exe
TARGET_PHASE_TRANS = $(BUILD_DIR)/phase_transition.exe
TARGET_PHASE_TRANS_IMPROVED = $(BUILD_DIR)/phase_transition_improved.exe

# Default target
all: directories $(TARGET_ISING) $(TARGET_BURNIN) $(TARGET_ENERGY_DIST) \
     $(TARGET_PARALLEL) $(TARGET_PHASE_TRANS) $(TARGET_PHASE_TRANS_IMPROVED)

# Create required directories
directories: $(BUILD_DIR) $(DATA_DIR) $(FIGURES_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(FIGURES_DIR):
	mkdir -p $(FIGURES_DIR)

# Build rules
$(TARGET_ISING): $(SRC_ISING_TESTS) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_BURNIN): $(SRC_BURNING) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_ENERGY_DIST): $(SRC_ENERGY_DIST) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_PARALLEL): $(SRC_PARALLEL) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_PHASE_TRANS): $(SRC_PHASE_TRANS) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_PHASE_TRANS_IMPROVED): $(SRC_PHASE_TRANS_IMPROVED) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Show compiler information
compiler:
	@echo "Using compiler: $(CXX)"
	@$(CXX) --version

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Remove generated data and figures
cleandata:
	rm -rf $(DATA_DIR)
	rm -rf $(FIGURES_DIR)

# Clean everything
cleanall: clean cleandata

# Phony targets
.PHONY: all clean cleandata distclean compiler directories