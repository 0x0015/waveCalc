OBJS = main.cpp.o \
       waveChamber2D.cpp.o \
       waveChamber3D.cpp.o \
       imageWriter.cpp.o \
       rawDataWriter.cpp.o

OUT	= main
SYCL    = acpp
CC      = gcc
CC_ACCEL = ccache
BUILD_CXX_FLAGS	 = -Wall -std=c++20 -g -O3 -march=native
BULID_CC_FLAGS   =
LINK_OPTS	 = 

all: $(OBJS)
	$(CC_ACCEL) $(SYCL) $(OBJS) $(BUILD_CXX_FLAGS) -g -o $(OUT) $(LINK_OPTS)

%.cpp.o: %.cpp
	$(CC_ACCEL) $(SYCL) $< $(BUILD_CXX_FLAGS) -c -o $@

%.c.o: %.c
	$(CC_ACCEL) $(SYCL) $< $(BUILD_CXX_FLAGS) -c -o $@

clean:
	rm -f $(OBJS) $(OUT)
	rm -f output.mp4
	rm -f outputImages/*.png
