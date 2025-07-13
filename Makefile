OBJS = waveSim/waveChamber2D.cpp.o \
       waveSim/waveChamber3D.cpp.o \
       waveSim/rawDataWriter.cpp.o \
       processOutput/rawDataLoader.cpp.o \
       processOutput/imageWriter.cpp.o \
       processOutput/fftProcess.cpp.o

MAIN_OBJS = waveSim_main.cpp.o \
	    processOutput_main.cpp.o

WAVESIM_OUT = waveSim_main
PROCESSOUTPUT_OUT = processOutput_main
SYCL    = acpp
CC      = gcc
CC_ACCEL = ccache
BUILD_CXX_FLAGS	 = -Wall -std=c++20 -g -O3 -flto -march=native -ffast-math
BULID_CC_FLAGS   =
LINK_OPTS	 =  -flto -lfftw3

all: $(OBJS) $(MAIN_OBJS) waveSim_main processOutput_main

waveSim_main: $(OBJS) waveSim_main.cpp.o
	$(CC_ACCEL) $(SYCL) $(OBJS) waveSim_main.cpp.o $(BUILD_CXX_FLAGS) -g -o $(WAVESIM_OUT) $(LINK_OPTS)

processOutput_main: $(OBJS) processOutput_main.cpp.o
	$(CC_ACCEL) $(SYCL) $(OBJS) processOutput_main.cpp.o $(BUILD_CXX_FLAGS) -g -o $(PROCESSOUTPUT_OUT) $(LINK_OPTS)

dpcpp: SYCL = /opt/intel/oneapi/compiler/latest/bin/icpx -fsycl -fsycl-targets=spir64_x86_64
dpcpp: all

%.cpp.o: %.cpp
	$(CC_ACCEL) $(SYCL) $< $(BUILD_CXX_FLAGS) -c -o $@

%.c.o: %.c
	$(CC_ACCEL) $(SYCL) $< $(BUILD_CXX_FLAGS) -c -o $@

clean:
	rm -f $(OBJS) $(MAIN_OBJS) $(WAVESIM_OUT) $(PROCESSOUTPUT_OUT)
	rm -f output.mp4
	rm -f outputImages/*.png
