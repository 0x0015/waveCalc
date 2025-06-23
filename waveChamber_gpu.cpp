#include "waveChamber.hpp"
#include <iostream>
#include "gpuLib/algorithm.hpp"
#include "waveChamberUCalc.hpp"

template<> void waveChamber<EXECUTION_MODE_GPU>::initStateDats(){
	for(auto& state : state_dats){
		auto gpuState = agpuUtil::device_vector<double>();
		gpuState.resize(partitions.x * partitions.y);
		agpuUtil::fill(gpuState, 0.0);
		state = gpuState;
	}
}

template<> void waveChamber<EXECUTION_MODE_GPU>::initChambers(std::span<const chamberDef> chambers){
	std::vector<chamberDef> cpuChambers;
	cpuChambers.resize(chambers.size());
	for(unsigned int i=0;i<chambers.size();i++){
		cpuChambers[i] = chambers[i];
		cpuChambers[i].size_internal = (cpuChambers[i].size / partitionSize).convert<unsigned int>();
		cpuChambers[i].pos_internal = (cpuChambers[i].pos / partitionSize).convert<unsigned int>();
	}
	auto gpuChambers = agpuUtil::device_vector<chamberDef>(cpuChambers);
	chamberDefs = gpuChambers;
}

template<> void waveChamber<EXECUTION_MODE_GPU>::printuVals(){
	agpuUtil::check_error(agpuDeviceSynchronize());
	auto copiedData = state_dats[currentStateNum].copy_to_host();
	auto accessor = array2DWrapper<double>(copiedData.data(), copiedData.size(), partitions);
	for(unsigned int i=0;i<accessor.size.x;i++){
		for(unsigned int j=0;j<accessor.size.y;j++){
			std::cout<<accessor[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
}

template<> void waveChamber<EXECUTION_MODE_GPU>::writeToImage(const std::string& filename, double expectedMax){
	agpuUtil::check_error(agpuDeviceSynchronize());
	const auto& copiedData = state_dats[currentStateNum].copy_to_host();
	imgWriter.createRequest(imageWriter::imageWriteRequest{copiedData, partitions, filename, expectedMax});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

template<> void waveChamber<EXECUTION_MODE_GPU>::writeRawData(){
	agpuUtil::check_error(agpuDeviceSynchronize());
	const auto& copiedData = state_dats[currentStateNum].copy_to_host();
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{copiedData});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

__global__ void step_kernel(double* newState_raw, const double* currentState_raw, const double* previousState_raw, unsigned int stateLength, vec2<unsigned int> partitions, double partitionSize, double dt, const waveChamber<EXECUTION_MODE_GPU>::chamberDef* chamberDefs, unsigned int chamberDefsLength){	
	const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid >= stateLength)
		return;

	unsigned int x = gid % partitions.x;
	unsigned int y = gid / partitions.x;

	if(x == 0 || x == partitions.x - 1 || y == 0 || y == partitions.y - 1)
		return;

	array2DWrapper_view<double> currentState(currentState_raw, stateLength, partitions);
	array2DWrapper_view<double> previousState(previousState_raw, stateLength, partitions);
	array2DWrapper<double> newState(newState_raw, stateLength, partitions);

	int chamberNum = -1;
	for(int chamberIndex=0;chamberIndex<(int)chamberDefsLength;chamberIndex++){
		if(chamberDefs[chamberIndex].isPointInChamber({x, y})){
			chamberNum = chamberIndex;
			break;
		}
	}
	if(chamberNum != -1){
		newState[{x, y}] = calculateUAtPos({x, y}, {currentState[{x, y}], currentState[{x+1, y}], currentState[{x-1, y}], currentState[{x, y+1}], currentState[{x, y-1}]}, previousState[{x, y}], partitionSize, chamberDefs[chamberNum].c, dt, chamberDefs[chamberNum].mu);
	}else{
		newState[{x, y}] = 0;
	}
}

template<> void waveChamber<EXECUTION_MODE_GPU>::step(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];

	//use block optimizer later
	constexpr unsigned int blockSize = 256;
	KERNEL_LAUNCH(step_kernel, gpuGridSize, gpuBlockSize)(state_dats[nextStateNum].data(), state_dats[currentStateNum].data(), state_dats[previousStateNum].data(), state_dats.front().size(), partitions, partitionSize, dt, chamberDefs.data(), chamberDefs.size());

	currentState = nextState;
	currentStateNum = nextStateNum;
}

__global__ void setSinglePoint_kernel(double* state_raw, unsigned int stateLength, vec2<unsigned int> partitions, vec2<unsigned int> point, double val){	
	const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid >= stateLength)
		return;

	unsigned int x = gid % partitions.x;
	unsigned int y = gid / partitions.x;

	array2DWrapper<double> state(state_raw, stateLength, partitions);

	if(point == vec2<unsigned int>{x, y})
		state[{x, y}] = val;
}

template<> void waveChamber<EXECUTION_MODE_GPU>::setSinglePoint(vec2<unsigned int> point, double val){
	agpuUtil::check_error(agpuDeviceSynchronize());
	constexpr unsigned int blockSize = 256;
	KERNEL_LAUNCH(setSinglePoint_kernel, state_dats.front().size()/blockSize+ (state_dats.front().size() % blockSize != 0), blockSize)(state_dats[currentStateNum].data(), state_dats[currentStateNum].size(), partitions, point, val);
}

template<> void waveChamber<EXECUTION_MODE_GPU>::calculateBestGpuOccupancy(){
	agpuUtil::check_error(agpuOccupancyMaxPotentialBlockSize(&gpuMinGridSize, &gpuBlockSize, step_kernel, 0, 0));
	gpuGridSize = (state_dats.front().size() + gpuBlockSize - 1)/gpuBlockSize;
}

template class waveChamber<EXECUTION_MODE_CPU>;
template class waveChamber<EXECUTION_MODE_GPU>;
