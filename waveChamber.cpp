#include "waveChamber.hpp"
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void waveChamber::init(vec2<double> s, double _dt, double _c, double _mu, unsigned int xPartitions, EXECUTION_MODE mode){
	dt = _dt;
	c = _c;
	mu = _mu;
	size = s;
	executionMode = mode;
	partitionSize = s.x / (double)xPartitions;
	double yPartitions = s.y / partitionSize;
	partitions = {xPartitions, (unsigned int)std::round(yPartitions)};
	switch (mode){
		case EXECUTION_MODE_CPU:
			initStateDats_cpu();
			break;
		case EXECUTION_MODE_GPU:
			initStateDats_gpu();
			break;
	}
	for(unsigned int i=0;i<states.size();i++){
		states[i].uVals = array2DWrapper<double>(state_dats[i]->data(), state_dats[i]->size(), partitions);
	}
	currentStateNum = 0;
	currentState = &states[currentStateNum];
}

void waveChamber::initStateDats_cpu(){
	for(auto& state : state_dats){
		auto cpuState = std::make_shared<cpuContainer<double>>();
		cpuState->cpuData.resize(partitions.x * partitions.y);
		state = cpuState;
	}
}

//based on https://www.csun.edu/~jb715473/math592c/wave2d.pdf

void waveChamber::step_cpu(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];


	const auto calcU = [&](unsigned int x, unsigned int y){
		double term1 = currentState->uVals[{x, y}] * (4.0-8.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
		double term2 = previousState->uVals[{x, y}] * (mu*dt-2.0) / (mu*dt+2.0);
		double term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
		double term3_pt2 = currentState->uVals[{x+1,y}] + currentState->uVals[{x-1,y}] + currentState->uVals[{x,y+1}] + currentState->uVals[{x, y-1}];
		double term3 = term3_pt1 * term3_pt2;
		return term1 + term2 + term3;
	};

	for(unsigned int i=1;i<partitions.x-1;i++){
		for(unsigned int j=1;j<partitions.y-1;j++){
			nextState->uVals[{i, j}] = calcU(i, j);
		}
	}

	currentState = nextState;
}

void waveChamber::step(){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			step_cpu();
			break;
		case EXECUTION_MODE_GPU:
			step_gpu();
			break;
	}
}

void waveChamber::printuVals(){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			printuVals_cpu();
			break;
		case EXECUTION_MODE_GPU:
			printuVals_gpu();
			break;
	}
}

void waveChamber::printuVals_cpu(){
	for(unsigned int i=0;i<currentState->uVals.size.x;i++){
		for(unsigned int j=0;j<currentState->uVals.size.y;j++){
			std::cout<<currentState->uVals[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
}

void waveChamber::writeToImage(const std::string& filename, double expectedMax){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			writeToImage_cpu(filename, expectedMax);
			break;
		case EXECUTION_MODE_GPU:
			writeToImage_gpu(filename, expectedMax);
			break;
	}
}

void waveChamber::writeToImage_cpu(const std::string& filename, double expectedMax){
	writeToImage_internals(filename, expectedMax, currentState->uVals);
}

void waveChamber::writeToImage_internals(const std::string& filename, double expectedMax, array2DWrapper<double> arrWr){
	std::vector<uint8_t> imageData(partitions.x * partitions.y*3);

	unsigned int x=0;
	unsigned int y=0;
	for(unsigned int i=0;i<imageData.size();i+=3){
		double val = arrWr[{x, y}];
		double scaled = (val) / expectedMax * 255;
		imageData[i+1] = 0;
		if(scaled < 0){
			imageData[i] = -scaled;
			imageData[i+2] = 0;
		}else{
			imageData[i] = 0;
			imageData[i+2] = scaled;
		}
		x++;
		if(x >= partitions.x){
			x = 0;
			y++;
		}
	}

	stbi_write_png(filename.c_str(), partitions.x, partitions.y, 3, imageData.data(), partitions.x * 3);
}

void waveChamber::setSinglePoint(vec2<unsigned int> point, double val){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			setSinglePoint_cpu(point, val);
			break;
		case EXECUTION_MODE_GPU:
			setSinglePoint_gpu(point, val);
			break;
	}
}

void waveChamber::setSinglePoint_cpu(vec2<unsigned int> point, double val){
	currentState->uVals[point] = val;
}


