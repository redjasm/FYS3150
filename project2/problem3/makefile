# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 $(shell pkg-config --cflags armadillo)

# Linker flags
LDFLAGS = $(shell pkg-config --libs armadillo)

# Target executable
TARGET = solution.exe

# Source file
SRC = solution.cpp

# Default target
all: $(TARGET)

# Build the target executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean the build (removes the executable)
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
