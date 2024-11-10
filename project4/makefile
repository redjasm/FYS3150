# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -I include $(shell pkg-config --cflags armadillo)

# Linker flags
LDFLAGS = $(shell pkg-config --libs armadillo)

# Target executable
TARGET = build/ising.exe

# Source files
SRC = src/main.cpp src/IsingModel.cpp

# Build directory
BUILD_DIR = build

# Default target
all: $(BUILD_DIR) $(TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build the target executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean the build (removes the executable and directory)
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean