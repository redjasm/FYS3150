# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -I include $(shell pkg-config --cflags armadillo)

# Linker flags
LDFLAGS = $(shell pkg-config --libs armadillo)

# Build directory
BUILD_DIR = build

# Source files
SRC_ISING = src/IsingModel.cpp
SRC_MAIN = src/main.cpp
SRC_BURNIN = src/burning_study.cpp

# Target executables
TARGET_MAIN = $(BUILD_DIR)/ising.exe
TARGET_BURNIN = $(BUILD_DIR)/burning.exe

# Default target
all: $(BUILD_DIR) $(TARGET_MAIN) $(TARGET_BURNIN)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build main executable
$(TARGET_MAIN): $(SRC_MAIN) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build burn-in study executable
$(TARGET_BURNIN): $(SRC_BURNIN) $(SRC_ISING)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Clean the build (removes the executable and directory)
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean