#include "gpuContainer.hpp"
#include "waveChamber.hpp"
#include <iostream>
#include "gpuLib/algorithm.hpp"
#include "waveChamberUCalc.hpp"

void waveChamber::initStateDats_gpu(){
	for(auto& state : state_dats){
		auto gpuState = std::make_shared<gpuContainer<double>>();
		gpuState->gpuData.resize(partitions.x * partitions.y);
		hipUtil::fill(gpuState->gpuData, 0.0);
		state = gpuState;
	}
}

void waveChamber::printuVals_gpu(){
	hipUtil::check_error(agpuDeviceSynchronize());
	auto copiedData = std::dynamic_pointer_cast<gpuContainer<double>>(state_dats[currentStateNum])->gpuData.copy_to_host();
	auto accessor = array2DWrapper<double>(copiedData.data(), copiedData.size(), partitions);
	for(unsigned int i=0;i<accessor.size.x;i++){
		for(unsigned int j=0;j<accessor.size.y;j++){
			std::cout<<accessor[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
}

void waveChamber::writeToImage_gpu(const std::string& filename, double expectedMax){
	hipUtil::check_error(agpuDeviceSynchronize());
	const auto& copiedData = std::dynamic_pointer_cast<gpuContainer<double>>(state_dats[currentStateNum])->gpuData.copy_to_host();
	imgWriter.createRequest(imageWriter::imageWriteRequest{copiedData, partitions, filename, expectedMax});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

void waveChamber::writeRawData_gpu(){
	hipUtil::check_error(agpuDeviceSynchronize());
	const auto& copiedData = std::dynamic_pointer_cast<gpuContainer<double>>(state_dats[currentStateNum])->gpuData.copy_to_host();
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{copiedData});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

__global__ void step_kernel(double* newState_raw, const double* currentState_raw, const double* previousState_raw, unsigned int stateLength, vec2<unsigned int> partitions, double partitionSize, double c, double dt, double mu){	
	const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid >= stateLength)
		return;

	unsigned int x = gid % partitions.x;
	unsigned int y = gid / partitions.x;

	if(x == 0 || x == partitions.x - 1 || y == 0 || y == partitions.y - 1)
		return;

	array2DWrapper_const<double> currentState(currentState_raw, stateLength, partitions);
	array2DWrapper_const<double> previousState(previousState_raw, stateLength, partitions);
	array2DWrapper<double> newState(newState_raw, stateLength, partitions);

	newState[{x, y}] = calculateUAtPos({x, y}, {currentState[{x, y}], currentState[{x+1, y}], currentState[{x-1, y}], currentState[{x, y+1}], currentState[{x, y-1}]}, previousState[{x, y}], partitionSize, c, dt, mu);
}

void waveChamber::step_gpu(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];

	//use block optimizer later
	constexpr unsigned int blockSize = 256;
	step_kernel<<<gpuGridSize, gpuBlockSize>>>(state_dats[nextStateNum]->data(), state_dats[currentStateNum]->data(), state_dats[previousStateNum]->data(), state_dats.front()->size(), partitions, partitionSize, c, dt, mu);

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

void waveChamber::setSinglePoint_gpu(vec2<unsigned int> point, double val){
	hipUtil::check_error(agpuDeviceSynchronize());
	constexpr unsigned int blockSize = 256;
	setSinglePoint_kernel<<<state_dats.front()->size()/blockSize+ (state_dats.front()->size() % blockSize != 0), blockSize>>>(state_dats[currentStateNum]->data(), state_dats[currentStateNum]->size(), partitions, point, val);
}

void waveChamber::calculateBestGpuOccupancy(){
	hipUtil::check_error(agpuOccupancyMaxPotentialBlockSize(&gpuMinGridSize, &gpuBlockSize, step_kernel, 0, 0));
	gpuGridSize = (state_dats.front()->size() + gpuBlockSize - 1)/gpuBlockSize;
}

