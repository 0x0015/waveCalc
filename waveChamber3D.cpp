#include "waveChamber3D.hpp"
#include <iostream>
#include "waveChamberUCalc.hpp"
#include <chrono>
#include <cmath>

template<EXECUTION_MODE exMode> void waveChamber3D<exMode>::init(vec<double, dim> s, double _dt, unsigned int xPartitions, std::span<const chamberDef<dim>> chambers, simulationEdgeMode simEdgeMode){
	dt = _dt;
	size = s;
	partitionSize = s.x / (double)xPartitions;
	double yPartitions = s.y / partitionSize;
	double zPartitions = s.z / partitionSize;
	partitions = {xPartitions, (unsigned int)std::round(yPartitions), (unsigned int)std::round(zPartitions)};
	edgeMode = simEdgeMode;
	initStateDats();
	initChambers(chambers);
	for(unsigned int i=0;i<states.size();i++){
		states[i].uVals = arrayNDWrapper<double, dim>(state_dats[i].data(), state_dats[i].size(), partitions);
	}
	currentStateNum = 0;
	currentState = &states[currentStateNum];
	if constexpr(exMode == EXECUTION_MODE_GPU)
		calculateBestGpuOccupancy();
}

template<> void waveChamber<EXECUTION_MODE_CPU, 3>::initStateDats(){
	for(auto& state : state_dats){
		auto cpuState = std::vector<double>(0.0);
		cpuState.resize(partitions.elementwiseProduct());
		state = cpuState;
	}
}

template<> void waveChamber<EXECUTION_MODE_CPU, 3>::initChambers(std::span<const chamberDef_t> chambers){
	auto cpuChambers = std::vector<chamberDef_t>();
	cpuChambers.resize(chambers.size());
	for(unsigned int i=0;i<chambers.size();i++){
		cpuChambers[i] = chambers[i];
		cpuChambers[i].size_internal = (cpuChambers[i].size / partitionSize).convert<unsigned int>();
		cpuChambers[i].pos_internal = (cpuChambers[i].pos / partitionSize).convert<unsigned int>();
	}
	chamberDefs = cpuChambers;
}

//based on https://www.csun.edu/~jb715473/math592c/wave2d.pdf

template<> void waveChamber<EXECUTION_MODE_CPU, 3>::step(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];

#pragma omp parallel for
	for(unsigned int i=1;i<partitions.x-1;i++){
		for(unsigned int j=1;j<partitions.y-1;j++){
			for(unsigned int k=1;k<partitions.z-1;k++){
				if(i == 0 || i == partitions.x - 1 || j == 0 || j == partitions.y - 1 || k == 0 || k == partitions.z - 1){
					switch(edgeMode){
						case REFLECT:
							continue;
						case VOID:
							nextState->uVals[{i, j, k}] = 0;
							continue;
					}
				}
				int chamberNum = -1;
				for(int chamberIndex=0;chamberIndex<(int)chamberDefs.size();chamberIndex++){
					if(chamberDefs[chamberIndex].isPointInChamber({i, j, k})){
						chamberNum = chamberIndex;
						break;
					}
				}
				if(chamberNum != -1){
					nextState->uVals[{i, j, k}] = calculateUAtPos3D({i, j, k}, {currentState->uVals[{i, j, k}], currentState->uVals[{i+1, j, k}], currentState->uVals[{i-1, j, k}], currentState->uVals[{i, j+1, k}], currentState->uVals[{i, j-1, k}], currentState->uVals[{i, j, k+1}], currentState->uVals[{i, j, k-1}]}, previousState->uVals[{i, j, k}], partitionSize, chamberDefs[chamberNum].c, dt, chamberDefs[chamberNum].mu);
				}else{
					nextState->uVals[{i, j, k}] = 0;
				}
			}
		}
	}

	currentState = nextState;
	currentStateNum = nextStateNum;
}

template<> void waveChamber<EXECUTION_MODE_CPU, 3>::printuVals(){
	for(unsigned int i=0;i<currentState->uVals.size.x;i++){
		for(unsigned int j=0;j<currentState->uVals.size.y;j++){
			std::cout<<currentState->uVals[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
}

template<> void waveChamber<EXECUTION_MODE_CPU, 3>::writeToImage(const std::string& filename, double expectedMax){
	//TODO!!
	//imgWriter.createRequest(imageWriter::imageWriteRequest{state_dats[currentStateNum], partitions, filename, expectedMax});
	//if(!imgWriter.threadsRun)
	//	imgWriter.processAllRequestsSynchronous();
}

template<> void waveChamber<EXECUTION_MODE_CPU, 3>::writeRawData(){
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{state_dats[currentStateNum]});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

template<> void waveChamber<EXECUTION_MODE_CPU, 3>::setSinglePoint(vec3<double> point, double val){
	currentState->uVals[(point / partitionSize).convert<unsigned int>()] = val;
}

template<EXECUTION_MODE exMode> void waveChamber3D<exMode>::setPointVarryingTimeFunction(vec3<double> point, std::function<double(double)> valWRTTimeGenerator){
	varryingTimeFuncs.push_back({point, valWRTTimeGenerator});
}

template<EXECUTION_MODE exMode> void waveChamber3D<exMode>::runSimulation(double time, double imageSaveInterval, double printRuntimeStatisticsInterval, double saveRawDataInterval){
	std::cout<<"Simulating for "<<time<<" seconds (dt = "<<dt<<", partition size = "<<partitionSize<<", execution mode = "<<std::array{"CPU", "GPU"}[(unsigned int)exMode]<<").  Chambers:"<<std::endl;
	std::vector<chamberDef<dim>> cpuChambers;
	if constexpr(exMode == EXECUTION_MODE_CPU)
		cpuChambers = chamberDefs;
	if constexpr(exMode == EXECUTION_MODE_GPU)
		cpuChambers = chamberDefs.copy_to_host();
	for(unsigned int i=0;i<cpuChambers.size();i++){
		const auto& ch = cpuChambers[i];
		std::cout<<"\tChamber "<<i<<": Pos = {"<<ch.pos.x<<", "<<ch.pos.y<<", "<<ch.pos.z<<"} (internally {"<<ch.pos_internal.x<<", "<<ch.pos_internal.y<<", "<<ch.pos_internal.z<<"}), Size = {"<<ch.size.x<<", "<<ch.size.y<<", "<<ch.size.z<<"} (internally {"<<ch.size_internal.x<<", "<<ch.size_internal.y<<", "<<ch.size_internal.z<<"}), Wave speed c = "<<ch.c<<", Damping μ = "<<ch.mu<<std::endl;
	}
	auto start = std::chrono::high_resolution_clock::now();
	auto start_time_t = std::chrono::system_clock::to_time_t(start);
	std::cout<<"Starting simulation at "<<std::ctime(&start_time_t);

	//if we're on the gpu, sure throw threads at making the image writing go faster (very very minor (if any) slowdown on the gpu side).
	//if we're on the cpu, we need all the threads we can get, so throwing threads at something else won't help at all.
	imgWriter.launchThreads(exMode == EXECUTION_MODE_GPU ? 4 : 1);

	if(saveRawDataInterval > 0){
		//TODO
		//rawWriter.createFile("rawOutput.grid2dD", partitions);
		rawWriter.launchThread();
	}

	double timeSinceLastSave = 0.0;
	double timeSinceLastStats = 0.0;
	double timeSinceLastRawWrite = 0.0;
	unsigned int imageSaveNum = 0;
	for(double currentlySimulatedTime = 0.0; currentlySimulatedTime < time; currentlySimulatedTime+=dt){
		for(const auto& [pos, func] : varryingTimeFuncs){
			setSinglePoint(pos, func(currentlySimulatedTime));
		}
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

template class waveChamber3D<EXECUTION_MODE_CPU>;
#if !defined(DISABLE_GPU_EXECUTION)
template class waveChamber3D<EXECUTION_MODE_GPU>;
#endif

