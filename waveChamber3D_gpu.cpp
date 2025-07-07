#include "waveChamber.hpp"
#include <iostream>
#include "gpuLib/algorithm.hpp"
#include "waveChamberUCalc.hpp"

template<> void waveChamber3D<EXECUTION_MODE_GPU>::initStateDats(){
	for(auto& state : state_dats){
		state.resize(partitions.elementwiseProduct());
		agpuUtil::fill(state, 0.0);
	}
}

template<> void waveChamber3D<EXECUTION_MODE_GPU>::initChambers(std::span<const chamberDef<3>> chambers){
	std::vector<chamberDef<3>> cpuChambers;
	cpuChambers.resize(chambers.size());
	for(unsigned int i=0;i<chambers.size();i++){
		cpuChambers[i] = chambers[i];
		cpuChambers[i].size_internal = (cpuChambers[i].size / partitionSize).convert<unsigned int>();
		cpuChambers[i].pos_internal = (cpuChambers[i].pos / partitionSize).convert<unsigned int>();
	}
	auto gpuChambers = agpuUtil::device_vector<chamberDef<3>>(cpuChambers);
	chamberDefs = gpuChambers;
}

template<> void waveChamber3D<EXECUTION_MODE_GPU>::printuVals(){
	agpuUtil::check_error(agpuDeviceSynchronize());
	auto copiedData = state_dats[currentStateNum].copy_to_host();
	auto accessor = array3DWrapper<double>(copiedData.data(), copiedData.size(), partitions);
	for(unsigned int i=0;i<accessor.size.x;i++){
		for(unsigned int j=0;j<accessor.size.y;j++){
			for(unsigned int k=0;k<accessor.size.z;k++){
				std::cout<<accessor[{i, j, k}]<<" ";
			}
			std::cout<<std::endl;
		}
	}
}

template<> void waveChamber3D<EXECUTION_MODE_GPU>::writeToImage(const std::string& filename, double expectedMax){
	agpuUtil::check_error(agpuDeviceSynchronize());
	//TODO
	//const auto& copiedData = state_dats[currentStateNum].copy_to_host();
	//imgWriter.createRequest(imageWriter::imageWriteRequest{copiedData, partitions, filename, expectedMax});
	//if(!imgWriter.threadsRun)
	//	imgWriter.processAllRequestsSynchronous();
}

template<> void waveChamber3D<EXECUTION_MODE_GPU>::writeRawData(){
	agpuUtil::check_error(agpuDeviceSynchronize());
	const auto& copiedData = state_dats[currentStateNum].copy_to_host();
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{copiedData});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

__global__ void step_kernel_3d(double* newState_raw, const double* currentState_raw, const double* previousState_raw, unsigned int stateLength, vec3<unsigned int> partitions, double partitionSize, double dt, const waveChamber3D<EXECUTION_MODE_GPU>::chamberDef_t* chamberDefs, unsigned int chamberDefsLength, simulationEdgeMode edgeMode){	
	const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid >= stateLength)
		return;

	unsigned int x = gid % partitions.x;
	unsigned int y = (gid / partitions.x) % partitions.y;
	unsigned int z = gid / (partitions.x * partitions.y);

	array3DWrapper_view<double> currentState(currentState_raw, stateLength, partitions);
	array3DWrapper_view<double> previousState(previousState_raw, stateLength, partitions);
	array3DWrapper<double> newState(newState_raw, stateLength, partitions);

	int chamberNum = -1;
	for(int chamberIndex=0;chamberIndex<(int)chamberDefsLength;chamberIndex++){
		if(chamberDefs[chamberIndex].isPointInChamber({x, y})){
			chamberNum = chamberIndex;
			break;
		}
	}

	if(chamberNum < 0){
		return;
	}

	switch(edgeMode){
		case REFLECT:
			if(x == 0){
				newState[{x, y, z}] = currentState[{x+1, y, z}];
				return;
			}
			if(x == partitions.x - 1){
				newState[{x, y, z}] = currentState[{x-1, y, z}];
				return;
			}
			if(y == 0){
				newState[{x, y, z}] = currentState[{x, y+1, z}];
				return;
			}
			if(y == partitions.y - 1){
				newState[{x, y, z}] = currentState[{x, y-1, z}];
				return;
			}
			if(z == 0){
				newState[{x, y, z}] = currentState[{x, y, z+1}];
				return;
			}
			if(z == partitions.z - 1){
				newState[{x, y, z}] = currentState[{x, y, z-1}];
				return;
			}
		case VOID:
			constexpr unsigned int edgeBoarder = 5;
			if(x < edgeBoarder || x >= partitions.x - edgeBoarder || y < edgeBoarder || y >= partitions.y - edgeBoarder){
				newState[{x, y, z}] = calculateUAtPos3D({x, y, z}, {currentState[{x, y, z}], x == partitions.x-1 ? 0 : currentState[{x+1, y, z}], x == 0 ? 0 : currentState[{x-1, y, z}], y == partitions.y-1 ? 0 : currentState[{x, y+1, z}], y == 0 ? 0 : currentState[{x, y-1, z}], z == partitions.z-1 ? 0 : currentState[{x, y, z+1}], z == 0 ? 0 : currentState[{x, y, z-1}]}, previousState[{x, y, z}], partitionSize, chamberDefs[chamberNum].c, dt, 10000);
				return;
			}
	}

	newState[{x, y, z}] = calculateUAtPos3D({x, y, z}, {currentState[{x, y, z}], currentState[{x+1, y, z}], currentState[{x-1, y, z}], currentState[{x, y+1, z}], currentState[{x, y-1, z}], currentState[{x, y, z+1}], currentState[{x, y, z-1}]}, previousState[{x, y, z}], partitionSize, chamberDefs[chamberNum].c, dt, chamberDefs[chamberNum].mu);
}

template<> void waveChamber3D<EXECUTION_MODE_GPU>::step(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];

	KERNEL_LAUNCH(step_kernel_3d, gpuGridSize, gpuBlockSize)(state_dats[nextStateNum].data(), state_dats[currentStateNum].data(), state_dats[previousStateNum].data(), state_dats.front().size(), partitions, partitionSize, dt, chamberDefs.data(), chamberDefs.size(), edgeMode);

	currentState = nextState;
	currentStateNum = nextStateNum;
}

__global__ void setSinglePoint_kernel_3d(double* state_raw, unsigned int index, double val){
	state_raw[index] = val;
}

template<> void waveChamber3D<EXECUTION_MODE_GPU>::setSinglePoint(vec3<double> point, double val){
	agpuUtil::check_error(agpuDeviceSynchronize());
	array3DWrapper<double> state(state_dats[currentStateNum].data(), state_dats[currentStateNum].size(), partitions);
	KERNEL_LAUNCH(setSinglePoint_kernel_3d, 1, 1)(state_dats[currentStateNum].data(), state.computeIndex((point / partitionSize).convert<unsigned int>()), val); //this is the gpu segfault!
}

template<> void waveChamber3D<EXECUTION_MODE_GPU>::calculateBestGpuOccupancy(){
	agpuUtil::check_error(agpuOccupancyMaxPotentialBlockSize(&gpuMinGridSize, &gpuBlockSize, step_kernel_3d, 0, 0));
	gpuGridSize = (state_dats.front().size() + gpuBlockSize - 1)/gpuBlockSize;
}

template class waveChamber3D<EXECUTION_MODE_GPU>;

