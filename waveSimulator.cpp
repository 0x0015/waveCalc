#include "waveSimulator.hpp"
#include <iostream>
#include "waveChamberUCalc.hpp"
#include <chrono>
#include <cmath>

void waveSimulator::init(const waveEnvironment& waveEnv, double dt_new){
	dt = dt_new;
	environment = waveEnv;
	if(executionMode == EXECUTION_MODE_CPU){
		cpuAccessor = waveEnvironmentAccessor{environment.geometry->data(), environment.geometry->size(), environment.state->data(), environment.state->size(), 0};
	}
}

void waveSimulator::step_cpu(){
	cpuAccessor.activeStateNum = cpuAccessor.getStateNumInNStates(1);

#pragma omp parallel for
	for(unsigned int i=0;i<cpuAccessor.stateLength;i++){
		auto point = cpuAccessor.getPosFromStateIndex(i);
		const auto& partition = cpuAccessor.geometry[cpuAccessor.getGeometryIndexFromStateIndex(i)];
		cpuAccessor.set(point.x, point.y, calculateUAtPos(stateMembers{cpuAccessor.get(point.x, point.y, 1), cpuAccessor.get(point.x+1, point.y, 1), cpuAccessor.get(point.x-1, point.y, 1), cpuAccessor.get(point.x, point.y+1, 1), cpuAccessor.get(point.x, point.y-1, 1)}, cpuAccessor.get(point.x, point.y, 2), partition.size, partition.material.c, dt, partition.material.mu));
	}
}

void waveSimulator::step(){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			step_cpu();
			break;
		case EXECUTION_MODE_GPU:
			step_gpu();
			break;
	}
}

void waveSimulator::printuVals(){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			printuVals_cpu();
			break;
		case EXECUTION_MODE_GPU:
			printuVals_gpu();
			break;
	}
}

void waveSimulator::writeRawData(){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			writeRawData_cpu();
			break;
		case EXECUTION_MODE_GPU:
			writeRawData_gpu();
			break;
	}
}

void waveSimulator::printuVals_cpu(){
	auto boundingBox = cpuAccessor.getBounds();
	for(unsigned int j=boundingBox.first.y;j<boundingBox.second.y;j++){
		for(unsigned int i=boundingBox.first.x;i<boundingBox.second.x;i++){
			std::cout<<cpuAccessor.get(i, j)<<" ";
		}
		std::cout<<std::endl;
	}
}

void waveSimulator::writeToImage(const std::string& filename, double expectedMax){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			writeToImage_cpu(filename, expectedMax);
			break;
		case EXECUTION_MODE_GPU:
			writeToImage_gpu(filename, expectedMax);
			break;
	}
}

void waveSimulator::writeToImage_cpu(const std::string& filename, double expectedMax){
	imgWriter.createRequest(imageWriter::imageWriteRequest{std::dynamic_pointer_cast<cpuContainer<double>>(state_dats[currentStateNum])->cpuData, partitions, filename, expectedMax});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

void waveSimulator::writeRawData_cpu(){
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{std::dynamic_pointer_cast<cpuContainer<double>>(state_dats[currentStateNum])->cpuData});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

void waveSimulator::setSinglePoint(vec2<unsigned int> point, double val){
	switch (executionMode){
		case EXECUTION_MODE_CPU:
			setSinglePoint_cpu(point, val);
			break;
		case EXECUTION_MODE_GPU:
			setSinglePoint_gpu(point, val);
			break;
	}
}

void waveSimulator::setSinglePoint_cpu(vec2<unsigned int> point, double val){
	cpuAccessor.set(point.x, point.y, val);
}

void waveSimulator::runSimulation(double time, double imageSaveInterval, double printRuntimeStatisticsInterval, double saveRawDataInterval){
	std::cout<<"Simulating for "<<time<<" seconds (dt = "<<dt<<")"<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	auto start_time_t = std::chrono::system_clock::to_time_t(start);
	std::cout<<"Starting simulation at "<<std::ctime(&start_time_t);

	//if we're on the gpu, sure throw threads at making the image writing go faster (very very minor (if any) slowdown on the gpu side).
	//if we're on the cpu, we need all the threads we can get, so throwing threads at something else won't help at all.
	imgWriter.launchThreads(executionMode == EXECUTION_MODE_GPU ? 4 : 1);

	if(saveRawDataInterval > 0){
		rawWriter.createFile("rawOutput.grid2dD", partitions);
		rawWriter.launchThread();
	}

	double timeSinceLastSave = 0.0;
	double timeSinceLastStats = 0.0;
	double timeSinceLastRawWrite = 0.0;
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
		if(saveRawDataInterval > 0 && timeSinceLastRawWrite >= saveRawDataInterval){
			writeRawData();
			timeSinceLastRawWrite -= saveRawDataInterval;
		}
		timeSinceLastSave += dt;
		timeSinceLastStats += dt;
		timeSinceLastRawWrite += dt;
	}
	//make sure we get that last image at the end, if we want want
	if(imageSaveInterval > 0 && timeSinceLastSave >= imageSaveInterval){
		writeToImage("outputImages/image" + std::to_string(imageSaveNum) + ".png", 1.0);
		imageSaveNum++;
		timeSinceLastSave -= imageSaveInterval;
	}
	if(saveRawDataInterval > 0 && timeSinceLastRawWrite >= saveRawDataInterval){
		writeRawData();
		timeSinceLastRawWrite -= saveRawDataInterval;
	}

	imgWriter.joinThread();
	if(saveRawDataInterval > 0)
		rawWriter.joinThreadAndClose();

	auto end = std::chrono::high_resolution_clock::now();
	auto end_time_t = std::chrono::system_clock::to_time_t(end);
	std::cout<<"Finished simulation at "<<std::ctime(&end_time_t)<<std::endl;
	std::chrono::duration<double> elapsed = end-start;
	std::cout<<"Elapsed time: "<<elapsed<<std::endl;
}

