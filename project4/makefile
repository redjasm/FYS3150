# Compiler
# CXX = g++-12

# If the OS is macOS, use g++-12
ifeq ($(shell uname), Darwin)
	CXX = g++-12
endif

# If the compiler is not found, use g++
ifeq (, $(shell which $(CXX)))
	CXX = g++
endif


# Compiler flags
CXXFLAGS = -std=c++17 -O3 -I include $(shell pkg-config --cflags armadillo) -fopenmp

# Linker flags
LDFLAGS = $(shell pkg-config --libs armadillo) -fopenmp

# Build directory
BUILD_DIR = build

# Source files
SRC_ISING = src/IsingModel.cpp
SRC_MAIN = src/main.cpp
SRC_BURNIN = src/burning_study.cpp
SRC_ENERGY_DIST = src/energy_distribution.cpp
SRC_PARALLEL = src/parallel_study.cpp
SRC_PHASE_TRANSITION = src/phase_transition.cpp

# Target executables
TARGET_MAIN = $(BUILD_DIR)/ising.exe
TARGET_BURNIN = $(BUILD_DIR)/burning.exe
TARGET_ENERGY_DIST = $(BUILD_DIR)/energy_dist.exe
TARGET_PARALLEL = $(BUILD_DIR)/parallel.exe
TARGET_PHASE_TRANSITION = $(BUILD_DIR)/phase_transition.exe

# Default target
all: $(BUILD_DIR) $(TARGET_MAIN) $(TARGET_BURNIN) $(TARGET_ENERGY_DIST) $(TARGET_PARALLEL) $(TARGET_PHASE_TRANSITION) compiler

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build main executable
$(TARGET_MAIN): $(SRC_MAIN) $(SRC_ISING) 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build burn-in study executable
$(TARGET_BURNIN): $(SRC_BURNIN) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build energy distribution executable
$(TARGET_ENERGY_DIST): $(SRC_ENERGY_DIST) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build parallel study executable
$(TARGET_PARALLEL): $(SRC_PARALLEL) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build phase transition study executable
$(TARGET_PHASE_TRANSITION): $(SRC_PHASE_TRANSITION) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

compiler:
	@echo "Compiler: $(CXX)"
	@$(CXX) --version

# Clean the build (removes the executable and directory)
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean