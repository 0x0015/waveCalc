#include "waveChamber.hpp"
#include <iostream>
#include "waveChamberUCalc.hpp"
#include <chrono>
#include <cmath>

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
	if(mode == EXECUTION_MODE_GPU)
		calculateBestGpuOccupancy();
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

	for(unsigned int i=1;i<partitions.x-1;i++){
		for(unsigned int j=1;j<partitions.y-1;j++){
			nextState->uVals[{i, j}] = calculateUAtPos({i, j}, {currentState->uVals[{i, j}], currentState->uVals[{i+1, j}], currentState->uVals[{i-1, j}], currentState->uVals[{i, j+1}], currentState->uVals[{i, j-1}]}, previousState->uVals[{i, j}], partitionSize, c, dt, mu);
		}
	}

	currentState = nextState;
	currentStateNum = nextStateNum;
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
	imgWriter.createRequest(imageWriter::imageWriteRequest{std::dynamic_pointer_cast<cpuContainer<double>>(state_dats[currentStateNum])->cpuData, partitions, filename, expectedMax});
	if(!imgWriter.threadRunning)
		imgWriter.processAllRequestsSynchronous();
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

void waveChamber::runSimulation(double time, double imageSaveInterval, double printRuntimeStatisticsInterval){
	std::cout<<"Simulating for "<<time<<" seconds (dt = "<<dt<<", c="<<c<<", mu="<<mu<<", partition size="<<partitionSize<<")"<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	auto start_time_t = std::chrono::system_clock::to_time_t(start);
	std::cout<<"Starting simulation at "<<std::ctime(&start_time_t);

	imgWriter.launchThread();

	double timeSinceLastSave = 0.0;
	double timeSinceLastStats = 0.0;
	unsigned int imageSaveNum = 0;
	for(double currentlySimulatedTime = 0.0; currentlySimulatedTime < time; currentlySimulatedTime+=dt){
		step();
		if(imageSaveInterval > 0 && timeSinceLastSave >= imageSaveInterval){
			writeToImage("outputImages/image" + std::to_string(imageSaveNum) + ".png", 1.0);
			imageSaveNum++;
			timeSinceLastSave -= imageSaveInterval;
		}
		if(printRuntimeStatisticsInterval > 0 && timeSinceLastStats >= printRuntimeStatisticsInterval){
			auto update = std::chrono::high_resolution_clock::now();
			auto update_time_t = std::chrono::system_clock::to_time_t(update);
			std::chrono::duration<double> elapsed = update-start;
			std::cout<<"Completed "<<currentlySimulatedTime<<" of "<<time<<" simulated seconds ("<<currentlySimulatedTime / time * 100.0<<"% complete) with an elapsed time of "<<elapsed<<" at "<<std::ctime(&update_time_t);
			timeSinceLastStats -= printRuntimeStatisticsInterval;
		}
		timeSinceLastSave += dt;
		timeSinceLastStats += dt;
	}
	//make sure we get that last image at the end, if we want want
	if(imageSaveInterval > 0 && timeSinceLastSave >= imageSaveInterval){
		writeToImage("outputImages/image" + std::to_string(imageSaveNum) + ".png", 1.0);
		imageSaveNum++;
		timeSinceLastSave -= imageSaveInterval;
	}

	imgWriter.joinThread();

	auto end = std::chrono::high_resolution_clock::now();
	auto end_time_t = std::chrono::system_clock::to_time_t(end);
	std::cout<<"Finished simulation at "<<std::ctime(&end_time_t)<<std::endl;
	std::chrono::duration<double> elapsed = end-start;
	std::cout<<"Elapsed time: "<<elapsed<<std::endl;
}

