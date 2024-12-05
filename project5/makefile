CXX = g++
CXXFLAGS = -std=c++17 -Wall -I./include -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -larmadillo

SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

TARGET = main

SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/matrix.cpp
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)