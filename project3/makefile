# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -I include $(shell pkg-config --cflags armadillo)

# Linker flags
LDFLAGS = $(shell pkg-config --libs armadillo)

# Target executable
TARGET = build/penning_trap.exe

# Source files
SRC = src/main.cpp src/Particle.cpp src/PenningTrap.cpp

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