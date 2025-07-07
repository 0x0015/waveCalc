GPU_OBJS = main.cpp.gpu.o \
	   waveChamber2D.cpp.gpu.o \
	   waveChamber3D.cpp.gpu.o \
	   waveChamber2D_gpu.cpp.gpu.o \
	   waveChamber3D_gpu.cpp.gpu.o \
	   imageWriter.cpp.gpu.o \
	   rawDataWriter.cpp.gpu.o

NOGPU_OBJS = main.cpp.o \
	     waveChamber2D.cpp.o \
	     waveChamber3D.cpp.o \
	     imageWriter.cpp.o \
	     rawDataWriter.cpp.o

OUT	= main
CXX	= g++
GPU_CXX = $(shell ./getGpuCompiler.sh)
CC      = gcc
CC_ACCEL = ccache
BUILD_CXX_FLAGS	 = -Wall -std=c++20 -g -O3 -flto -fopenmp
BUILD_GPU_CXX_FLAGS = -std=c++20 -g -flto -O3 $(shell ./getGpuCompilerArgs.sh)
BULID_CC_FLAGS   =
LINK_OPTS	 = 

all: $(GPU_OBJS)
	$(GPU_CXX) $(GPU_OBJS) -fopenmp -g -o $(OUT) $(LINK_OPTS)

nogpu: BUILD_CXX_FLAGS += -DDISABLE_GPU_EXECUTION
nogpu: $(NOGPU_OBJS)
	$(CXX) $(NOGPU_OBJS) -fopenmp -g -o $(OUT) $(LINK_OPTS)

%.cpp.gpu.o: %.cpp
	$(CC_ACCEL) $(GPU_CXX) $< $(BUILD_GPU_CXX_FLAGS) -c -o $@

%.cpp.o: %.cpp
	$(CC_ACCEL) $(CXX) $< $(BUILD_CXX_FLAGS) -c -o $@

%.c.o: %.c
	$(CC_ACCEL) $(CXX) $< $(BUILD_CXX_FLAGS) -c -o $@

clean:
	rm -f $(OBJS) $(GPU_OBJS) $(OUT)
	rm -f output.mp4
	rm -f outputImages/*.png
