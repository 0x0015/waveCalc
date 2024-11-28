#include "gpuContainer.hpp"
#include "waveChamber.hpp"
#include <iostream>
#include "gpuLib/algorithm.hpp"

void waveChamber::initStateDats_gpu(){
	for(auto& state : state_dats){
		auto gpuState = std::make_shared<gpuContainer<double>>();
		gpuState->gpuData.resize(partitions.x * partitions.y);
		hipUtil::fill(gpuState->gpuData, 0.0);
		state = gpuState;
	}
}

void waveChamber::printuVals_gpu(){
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
	auto copiedData = std::dynamic_pointer_cast<gpuContainer<double>>(state_dats[currentStateNum])->gpuData.copy_to_host();
	auto accessor = array2DWrapper<double>(copiedData.data(), copiedData.size(), partitions);
	writeToImage_internals(filename, expectedMax, accessor);
}

__global__ void doStep(double* newState_raw, const double* currentState_raw, const double* previousState_raw, unsigned int stateLength, vec2<unsigned int> partitions, double partitionSize, double c, double dt, double mu){	
	const auto gid = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int x = gid % partitions.x;
	unsigned int y = gid / partitions.x;

	if(x == 0 || x == partitions.x - 1 || y == 0 || y == partitions.y - 1)
		return;

	array2DWrapper_const<double> currentState(currentState_raw, stateLength, partitions);
	array2DWrapper_const<double> previousState(previousState_raw, stateLength, partitions);
	array2DWrapper<double> newState(newState_raw, stateLength, partitions);

	double term1 = currentState[{x, y}] * (4.0-8.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term2 = previousState[{x, y}] * (mu*dt-2.0) / (mu*dt+2.0);
	double term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term3_pt2 = currentState[{x+1,y}] + currentState[{x-1,y}] + currentState[{x,y+1}] + currentState[{x, y-1}];
	double term3 = term3_pt1 * term3_pt2;

	newState[{x, y}] = term1 + term2 + term3;
}

void waveChamber::step_gpu(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];

	//use block optimizer later
	constexpr unsigned int blockSize = 256;
	doStep<<<state_dats.front()->size()/blockSize+ (state_dats.front()->size() % blockSize != 0), blockSize>>>(state_dats[nextStateNum]->data(), state_dats[currentStateNum]->data(), state_dats[previousStateNum]->data(), state_dats.front()->size(), partitions, partitionSize, c, dt, mu);

	currentState = nextState;
}

__global__ void setSinglePoint_kernel(double* state_raw, unsigned int stateLength, vec2<unsigned int> partitions, vec2<unsigned int> point, double val){	
	const auto gid = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int x = gid % partitions.x;
	unsigned int y = gid / partitions.x;

	array2DWrapper<double> state(state_raw, stateLength, partitions);

	if(point == vec2<unsigned int>{x, y})
		state[{x, y}] = val;
}

void waveChamber::setSinglePoint_gpu(vec2<unsigned int> point, double val){
	constexpr unsigned int blockSize = 256;
	setSinglePoint_kernel<<<state_dats.front()->size()/blockSize+ (state_dats.front()->size() % blockSize != 0), blockSize>>>(state_dats[currentStateNum]->data(), state_dats[currentStateNum]->size(), partitions, point, val);
}

