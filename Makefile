OBJS	= main.cpp.o \
	  waveChamber.cpp.o \
	  imageWriter.cpp.o \
	  rawDataWriter.cpp.o \


GPU_OBJS = waveChamber_gpu.cpp.gpu.o

OUT	= main
CXX	= g++
GPU_CXX = $(shell ./getGpuCompiler.sh)
CC      = gcc
CC_ACCEL = ccache
BUILD_CXX_FLAGS	 = -Wall -std=c++20 -g -fopenmp
BUILD_GPU_CXX_FLAGS = -std=c++20 -g $(shell ./getGpuCompilerArgs.sh)
BULID_CC_FLAGS   =
LINK_OPTS	 = 

all: $(OBJS) $(GPU_OBJS)
	$(GPU_CXX) $(OBJS) $(GPU_OBJS) -fopenmp -g -o $(OUT) $(LINK_OPTS)

%.cpp.gpu.o: %.cpp
	$(CC_ACCEL) $(GPU_CXX) $< $(BUILD_GPU_CXX_FLAGS) -c -o $@

%.cpp.o: %.cpp
	$(CC_ACCEL) $(CXX) $< $(BUILD_CXX_FLAGS) -c -o $@

%.c.o: %.c
	$(CC_ACCEL) $(CXX) $< $(BUILD_CXX_FLAGS) -c -o $@

clean:
	rm -f $(OBJS) $(GPU_OBJS) $(OUT)
