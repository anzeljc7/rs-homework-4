HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

BIN_DIR?= ./bin
SRC_FILES=$(wildcard *.cpp)
BIN_FILES=$(patsubst %.cpp,$(BIN_DIR)/%.bin,$(SRC_FILES))

all: $(BIN_FILES)

$(BIN_DIR)/%.bin: %.cpp $(BIN_DIR)
	$(HIPCC) $(CXXFLAGS) $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
