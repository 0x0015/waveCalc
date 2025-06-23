#include "waveChamber.hpp"
#include <iostream>
#include "waveChamberUCalc.hpp"
#include <chrono>
#include <cmath>

template<EXECUTION_MODE exMode> void waveChamber<exMode>::init(vec2<double> s, double _dt, unsigned int xPartitions, std::span<const chamberDef> chambers){
	dt = _dt;
	size = s;
	partitionSize = s.x / (double)xPartitions;
	double yPartitions = s.y / partitionSize;
	partitions = {xPartitions, (unsigned int)std::round(yPartitions)};
	initStateDats();
	initChambers(chambers);
	for(unsigned int i=0;i<states.size();i++){
		states[i].uVals = array2DWrapper<double>(state_dats[i].data(), state_dats[i].size(), partitions);
	}
	currentStateNum = 0;
	currentState = &states[currentStateNum];
	if constexpr(exMode == EXECUTION_MODE_GPU)
		calculateBestGpuOccupancy();
}

template<> void waveChamber<EXECUTION_MODE_CPU>::initStateDats(){
	for(auto& state : state_dats){
		auto cpuState = std::vector<double>(0.0);
		cpuState.resize(partitions.x * partitions.y);
		state = cpuState;
	}
}

template<> void waveChamber<EXECUTION_MODE_CPU>::initChambers(std::span<const chamberDef> chambers){
	auto cpuChambers = std::vector<chamberDef>();
	cpuChambers.resize(chambers.size());
	for(unsigned int i=0;i<chambers.size();i++){
		cpuChambers[i] = chambers[i];
		cpuChambers[i].size_internal = (cpuChambers[i].size / partitionSize).convert<unsigned int>();
		cpuChambers[i].pos_internal = (cpuChambers[i].pos / partitionSize).convert<unsigned int>();
	}
	chamberDefs = cpuChambers;
}

//based on https://www.csun.edu/~jb715473/math592c/wave2d.pdf

template<> void waveChamber<EXECUTION_MODE_CPU>::step(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];

#pragma omp parallel for
	for(unsigned int i=1;i<partitions.x-1;i++){
		for(unsigned int j=1;j<partitions.y-1;j++){
			int chamberNum = -1;
			for(int chamberIndex=0;chamberIndex<(int)chamberDefs.size();chamberIndex++){
				if(chamberDefs[chamberIndex].isPointInChamber({i, j})){
					chamberNum = chamberIndex;
					break;
				}
			}
			if(chamberNum != -1){
				nextState->uVals[{i, j}] = calculateUAtPos({i, j}, {currentState->uVals[{i, j}], currentState->uVals[{i+1, j}], currentState->uVals[{i-1, j}], currentState->uVals[{i, j+1}], currentState->uVals[{i, j-1}]}, previousState->uVals[{i, j}], partitionSize, chamberDefs[chamberNum].c, dt, chamberDefs[chamberNum].mu);
			}else{
				nextState->uVals[{i, j}] = 0;
			}
		}
	}

	currentState = nextState;
	currentStateNum = nextStateNum;
}

template<> void waveChamber<EXECUTION_MODE_CPU>::printuVals(){
	for(unsigned int i=0;i<currentState->uVals.size.x;i++){
		for(unsigned int j=0;j<currentState->uVals.size.y;j++){
			std::cout<<currentState->uVals[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
}

template<> void waveChamber<EXECUTION_MODE_CPU>::writeToImage(const std::string& filename, double expectedMax){
	imgWriter.createRequest(imageWriter::imageWriteRequest{state_dats[currentStateNum], partitions, filename, expectedMax});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

template<> void waveChamber<EXECUTION_MODE_CPU>::writeRawData(){
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{state_dats[currentStateNum]});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

template<> void waveChamber<EXECUTION_MODE_CPU>::setSinglePoint(vec2<unsigned int> point, double val){
	currentState->uVals[point] = val;
}

template<EXECUTION_MODE exMode> void waveChamber<exMode>::runSimulation(double time, double imageSaveInterval, double printRuntimeStatisticsInterval, double saveRawDataInterval){
	std::cout<<"Simulating for "<<time<<" seconds (dt = "<<dt<<", partition size="<<partitionSize<<")"<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	auto start_time_t = std::chrono::system_clock::to_time_t(start);
	std::cout<<"Starting simulation at "<<std::ctime(&start_time_t);

	//if we're on the gpu, sure throw threads at making the image writing go faster (very very minor (if any) slowdown on the gpu side).
	//if we're on the cpu, we need all the threads we can get, so throwing threads at something else won't help at all.
	imgWriter.launchThreads(exMode == EXECUTION_MODE_GPU ? 4 : 1);

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

template class waveChamber<EXECUTION_MODE_CPU>;
#if !defined(DISABLE_GPU_EXECUTION)
template class waveChamber<EXECUTION_MODE_GPU>;
#endif
