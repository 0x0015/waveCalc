OBJS	= main.cpp.o \
	  waveChamber.cpp.o

GPU_OBJS = gpu_example.cpp.gpu.o

OUT	= main
CXX	= g++
GPU_CXX = $(shell ./getGpuCompiler.sh)
CC      = gcc
BUILD_CXX_FLAGS	 = -Wall -std=c++20 -g
BULID_CC_FLAGS   =
LINK_OPTS	 = 

all: $(OBJS) $(GPU_OBJS)
	$(GPU_CXX) $(OBJS) $(GPU_OBJS) -o $(OUT) $(LINK_OPTS)

%.cpp.gpu.o: %.cpp
	$(GPU_CXX) $< $(BUILD_CXX_FLAGS) -c -o $@

%.cpp.o: %.cpp
	$(CXX) $< $(BUILD_CXX_FLAGS) -c -o $@

%.c.o: %.c
	$(CXX) $< $(BUILD_CXX_FLAGS) -c -o $@

clean:
	rm -f $(OBJS) $(GPU_OBJS) $(OUT)
