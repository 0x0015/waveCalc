#include "waveChamber3D.hpp"
#include <iostream>
#include <cmath>
#include "waveChamberUCalc.hpp"

template<typename float_t> void waveChamber3D<float_t>::init(sycl::vec<float_t, dim> s, float_t _dt, unsigned int xPartitions, std::span<const chamberDef_t> chambers, simulationEdgeMode simEdgeMode){
	dt = _dt;
	size = s;
	partitionSize = s.x() / (float_t)xPartitions;
	float_t yPartitions = s.y() / partitionSize;
	float_t zPartitions = s.z() / partitionSize;
	partitions = {xPartitions, (unsigned int)std::round(yPartitions), (unsigned int)std::round(zPartitions)};
	edgeMode = simEdgeMode;
	initStateDats();
	initChambers(chambers);
	for(unsigned int i=0;i<states.size();i++){
		states[i].uVals = arrayNDWrapper<float_t, dim>(state_dats[i].data(), state_dats[i].size(), partitions);
	}
	currentStateNum = 0;
	currentState = &states[currentStateNum];
}

template<typename float_t> void waveChamber3D<float_t>::initStateDats(){
	for(auto& state : state_dats){
		state.resize(partitions.x() * partitions.y() * partitions.z());
		q.fill<float_t>(state.data(), 0.0, state.size()).wait();
	}
}

template<typename float_t> void waveChamber3D<float_t>::initChambers(std::span<const chamberDef_t> chambers){
	std::vector<chamberDef_t> cpuChambers;
	cpuChambers.resize(chambers.size());
	for(unsigned int i=0;i<chambers.size();i++){
		cpuChambers[i] = chambers[i];
		cpuChambers[i].size_internal = (cpuChambers[i].size / partitionSize).template convert<unsigned int>();
		cpuChambers[i].pos_internal = (cpuChambers[i].pos / partitionSize).template convert<unsigned int>();
	}
	auto gpuChambers = deviceVector<chamberDef_t>(q, cpuChambers);
	chamberDefs = gpuChambers;
}

template<typename float_t> void waveChamber3D<float_t>::printuVals(){
	/*
	auto copiedData = state_dats[currentStateNum].copy_to_host();
	auto accessor = array2DWrapper<float_t>(copiedData.data(), copiedData.size(), partitions);
	for(unsigned int i=0;i<accessor.size.x();i++){
		for(unsigned int j=0;j<accessor.size.y();j++){
			std::cout<<accessor[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
	*/
}

template<typename float_t> void waveChamber3D<float_t>::writeRawData(){
	const auto& copiedData = state_dats[currentStateNum].copy_to_host();
	/*
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{copiedData});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
	*/
}

template<typename float_t> void waveChamber3D<float_t>::step(){
	q.submit([&](sycl::handler& handler){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;

	float_t* currentState_raw = state_dats[currentStateNum].data();
	unsigned int currentState_length = state_dats[currentStateNum].size();
	float_t* previousState_raw = state_dats[previousStateNum].data();
	unsigned int previousState_length = state_dats[previousStateNum].size();
	float_t* nextState_raw = state_dats[nextStateNum].data();
	unsigned int nextState_length = state_dats[nextStateNum].size();
	unsigned int chamberDefsLength = chamberDefs.size();
	chamberDef_t* chamberDefs = this->chamberDefs.data();
	simulationEdgeMode edgeMode = this->edgeMode;
	auto partitions = this->partitions;
	auto partitionSize = this->partitionSize;
	auto dt = this->dt;
	handler.parallel_for<class waveChamber2D_step>(sycl::range<3>(partitions.x(), partitions.y(), partitions.z()), [=](sycl::id<3> id) {
		array3DWrapper_view<float_t> currentState(currentState_raw, currentState_length, partitions);
		array3DWrapper_view<float_t> previousState(previousState_raw, previousState_length, partitions);
		array3DWrapper<float_t> newState(nextState_raw, nextState_length, partitions);
		unsigned int x = id.get(0);
		unsigned int y = id.get(1);
		unsigned int z = id.get(2);
		int chamberNum = -1;
		for(int chamberIndex=0;chamberIndex<(int)chamberDefsLength;chamberIndex++){
			if(chamberDefs[chamberIndex].isPointInChamber({x, y, z})){
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
				if(x == partitions.x() - 1){
					newState[{x, y, z}] = currentState[{x-1, y, z}];
					return;
				}
				if(y == 0){
					newState[{x, y, z}] = currentState[{x, y+1, z}];
					return;
				}
				if(y == partitions.y() - 1){
					newState[{x, y, z}] = currentState[{x, y-1, z}];
					return;
				}
				if(z == 0){
					newState[{x, y, z}] = currentState[{x, y, z+1}];
					return;
				}
				if(z == partitions.z() - 1){
					newState[{x, y, z}] = currentState[{x, y, z-1}];
					return;
				}
			case VOID:
				constexpr unsigned int edgeBoarder = 5;
				if(x < edgeBoarder || x >= partitions.x() - edgeBoarder || y < edgeBoarder || y >= partitions.y() - edgeBoarder || z < edgeBoarder || z >= partitions.z() - edgeBoarder){
					newState[{x, y, z}] = calculateUAtPos3D<float_t>({currentState[{x, y, z}], x == partitions.x()-1 ? 0 : currentState[{x+1, y, z}], x == 0 ? 0 : currentState[{x-1, y, z}], y == partitions.y()-1 ? 0 : currentState[{x, y+1, z}], y == 0 ? 0 : currentState[{x, y-1, z}], z == partitions.z()-1 ? 0 : currentState[{x, y, z+1}], z == 0 ? 0 : currentState[{x, y, z-1}]}, previousState[{x, y, z}], partitionSize, chamberDefs[chamberNum].c, dt, 10000);
					return;
				}
		}

		newState[{x, y, z}] = calculateUAtPos3D<float_t>({currentState[{x, y, z}], currentState[{x+1, y, z}], currentState[{x-1, y, z}], currentState[{x, y+1, z}], currentState[{x, y-1, z}], currentState[{x, y, z+1}], currentState[{x, y, z-1}]}, previousState[{x, y, z}], partitionSize, chamberDefs[chamberNum].c, dt, chamberDefs[chamberNum].mu);
	});

	//currentState = nextState;
	currentStateNum = nextStateNum;
	});
}

template<typename float_t> void waveChamber3D<float_t>::setSinglePoint(sycl::vec<float_t, 3> point, float_t val){
	array3DWrapper<float_t> state(state_dats[currentStateNum].data(), state_dats[currentStateNum].size(), partitions);
	auto convertedCoords = point / partitionSize;
	unsigned int index = state.computeIndex((unsigned int)convertedCoords.x(), (unsigned int)convertedCoords.y(), (unsigned int)convertedCoords.z());
	q.single_task<class setSinglePoind_2d>([=](){
		state.data[index] = val;
	});
}

template<typename float_t> void waveChamber3D<float_t>::setPointVarryingTimeFunction(sycl::vec<float_t, 3> point, std::function<float_t(float_t)> valWRTTimeGenerator){
	varryingTimeFuncs.push_back({point, valWRTTimeGenerator});
}

template<typename float_t> void waveChamber3D<float_t>::runSimulation(float_t time, float_t printRuntimeStatisticsInterval, float_t saveRawDataInterval){
	std::cout<<"Simulating for "<<time<<" seconds (dt = "<<dt<<", partition size = "<<partitionSize<<" execution device = ";
	if(q.get_device().is_cpu())
		std::cout<<"[CPU] ";
	if(q.get_device().is_gpu())
		std::cout<<"[GPU] ";
	std::cout<<q.get_device().get_info<sycl::info::device::name>();
	std::cout<<").  Chambers:"<<std::endl;
	std::vector<chamberDef_t> cpuChambers;
	cpuChambers = chamberDefs.copy_to_host();
	for(unsigned int i=0;i<cpuChambers.size();i++){
		const auto& ch = cpuChambers[i];
		std::cout<<"\tChamber "<<i<<": Pos = {"<<ch.pos.x()<<", "<<ch.pos.y()<<", "<<ch.pos.z()<<"} (internally {"<<ch.pos_internal.x()<<", "<<ch.pos_internal.y()<<", "<<ch.pos_internal.z()<<"}), Size = {"<<ch.size.x()<<", "<<ch.size.y()<<", "<<ch.size.z()<<"} (internally {"<<ch.size_internal.x()<<", "<<ch.size_internal.y()<<", "<<ch.size_internal.z()<<"}), Wave speed c = "<<ch.c<<", Damping μ = "<<ch.mu<<std::endl;
	}
	auto start = std::chrono::high_resolution_clock::now();
	auto start_time_t = std::chrono::system_clock::to_time_t(start);
	std::cout<<"Starting simulation at "<<std::ctime(&start_time_t);

	//if we're on the gpu, sure throw threads at making the image writing go faster (very very minor (if any) slowdown on the gpu side).
	//if we're on the cpu, we need all the threads we can get, so throwing threads at something else won't help at all.
	unsigned int imgWriterThreads = 4;
	{
		sycl::queue q;//eventually changed to unified queue (so I know ahead of time exactly what device was chosen)
		if(q.get_device().is_cpu())
			imgWriterThreads = 1;
	}

	if(saveRawDataInterval > 0){
		//rawWriter.createFile("rawOutput.grid2dD", partitions);
		rawWriter.launchThread();
	}

	float_t timeSinceLastSave = 0.0;
	float_t timeSinceLastStats = 0.0;
	float_t timeSinceLastRawWrite = 0.0;
	unsigned int imageSaveNum = 0;
	for(float_t currentlySimulatedTime = 0.0; currentlySimulatedTime < time; currentlySimulatedTime+=dt){
		for(const auto& [pos, func] : varryingTimeFuncs){
			setSinglePoint(pos, func(currentlySimulatedTime));
		}
		step();
		if(printRuntimeStatisticsInterval > 0 && timeSinceLastStats >= printRuntimeStatisticsInterval){
			auto update = std::chrono::high_resolution_clock::now();
			auto update_time_t = std::chrono::system_clock::to_time_t(update);
			std::chrono::duration<float_t> elapsed = update-start;
			std::cout<<"Completed "<<currentlySimulatedTime<<" of "<<time<<" simulated seconds ("<<currentlySimulatedTime / time * 100.0<<"% complete) with an elapsed time of "<<elapsed.count()<<" at "<<std::ctime(&update_time_t);
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
	if(saveRawDataInterval > 0 && timeSinceLastRawWrite >= saveRawDataInterval){
		writeRawData();
		timeSinceLastRawWrite -= saveRawDataInterval;
	}

	if(saveRawDataInterval > 0)
		rawWriter.joinThreadAndClose();

	auto end = std::chrono::high_resolution_clock::now();
	auto end_time_t = std::chrono::system_clock::to_time_t(end);
	std::cout<<"Finished simulation at "<<std::ctime(&end_time_t)<<std::endl;
	std::chrono::duration<float_t> elapsed = end-start;
	std::cout<<"Elapsed time: "<<elapsed.count()<<std::endl;
}

template class waveChamber3D<float>;
template class waveChamber3D<double>;

