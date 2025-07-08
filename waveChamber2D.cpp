#include "waveChamber2D.hpp"
#include <iostream>
#include "waveChamberUCalc.hpp"

void waveChamber2D::init(sycl::vec<double, dim> s, double _dt, unsigned int xPartitions, std::span<const chamberDef<dim>> chambers, simulationEdgeMode simEdgeMode){
	dt = _dt;
	size = s;
	partitionSize = s.x() / (double)xPartitions;
	double yPartitions = s.y() / partitionSize;
	partitions = {xPartitions, (unsigned int)std::round(yPartitions)};
	edgeMode = simEdgeMode;
	initStateDats();
	initChambers(chambers);
	for(unsigned int i=0;i<states.size();i++){
		states[i].uVals = arrayNDWrapper<double, dim>(state_dats[i].data(), state_dats[i].size(), partitions);
	}
	currentStateNum = 0;
	currentState = &states[currentStateNum];
}

void waveChamber2D::initStateDats(){
	for(auto& state : state_dats){
		state.resize(partitions.x() * partitions.y());
		q.fill(state.data(), 0.0, state.size());
		q.wait();
	}
}

void waveChamber2D::initChambers(std::span<const chamberDef_t> chambers){
	std::vector<chamberDef_t> cpuChambers;
	cpuChambers.resize(chambers.size());
	for(unsigned int i=0;i<chambers.size();i++){
		cpuChambers[i] = chambers[i];
		cpuChambers[i].size_internal = (cpuChambers[i].size / partitionSize).convert<unsigned int>();
		cpuChambers[i].pos_internal = (cpuChambers[i].pos / partitionSize).convert<unsigned int>();
	}
	auto gpuChambers = deviceVector<chamberDef_t>(q, cpuChambers);
	chamberDefs = gpuChambers;
}

void waveChamber2D::printuVals(){
	auto copiedData = state_dats[currentStateNum].copy_to_host();
	auto accessor = array2DWrapper<double>(copiedData.data(), copiedData.size(), partitions);
	for(unsigned int i=0;i<accessor.size.x();i++){
		for(unsigned int j=0;j<accessor.size.y();j++){
			std::cout<<accessor[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
}

void waveChamber2D::writeToImage(const std::string& filename, double expectedMax){
	const auto& copiedData = state_dats[currentStateNum].copy_to_host();
	imgWriter.createRequest(imageWriter::imageWriteRequest{copiedData, partitions, filename, expectedMax});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

void waveChamber2D::writeRawData(){
	const auto& copiedData = state_dats[currentStateNum].copy_to_host();
	rawWriter.createRequest(rawDataWriter::rawWriteRequest{copiedData});
	if(!imgWriter.threadsRun)
		imgWriter.processAllRequestsSynchronous();
}

void waveChamber2D::step(){
	q.submit([&](sycl::handler& handler){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;

	double* currentState_raw = state_dats[currentStateNum].data();
	unsigned int currentState_length = state_dats[currentStateNum].size();
	double* previousState_raw = state_dats[previousStateNum].data();
	unsigned int previousState_length = state_dats[previousStateNum].size();
	double* nextState_raw = state_dats[nextStateNum].data();
	unsigned int nextState_length = state_dats[nextStateNum].size();
	unsigned int chamberDefsLength = chamberDefs.size();
	chamberDef_t* chamberDefs = this->chamberDefs.data();
	simulationEdgeMode edgeMode = this->edgeMode;
	auto partitions = this->partitions;
	auto partitionSize = this->partitionSize;
	auto dt = this->dt;
	handler.parallel_for<class waveChamber2D_step>(sycl::range<2>(partitions.x(), partitions.y()), [=](sycl::id<2> id) {
		array2DWrapper_view<double> currentState(currentState_raw, currentState_length, partitions);
		array2DWrapper_view<double> previousState(previousState_raw, previousState_length, partitions);
		array2DWrapper<double> newState(nextState_raw, nextState_length, partitions);
		unsigned int x = id.get(0);
		unsigned int y = id.get(1);
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
					newState[{x, y}] = currentState[{x+1, y}];
					return;
				}
				if(x == partitions.x() - 1){
					newState[{x, y}] = currentState[{x-1, y}];
					return;
				}
				if(y == 0){
					newState[{x, y}] = currentState[{x, y+1}];
					return;
				}
				if(y == partitions.y() - 1){
					newState[{x, y}] = currentState[{x, y-1}];
					return;
				}
			case VOID:
				constexpr unsigned int edgeBoarder = 5;
				if(x < edgeBoarder || x >= partitions.x() - edgeBoarder || y < edgeBoarder || y >= partitions.y() - edgeBoarder){
					newState[{x, y}] = calculateUAtPos2D({currentState[{x, y}], x == partitions.x()-1 ? 0 : currentState[{x+1, y}], x == 0 ? 0 : currentState[{x-1, y}], y == partitions.y()-1 ? 0 : currentState[{x, y+1}], y == 0 ? 0 : currentState[{x, y-1}]}, previousState[{x, y}], partitionSize, chamberDefs[chamberNum].c, dt, 10000);
					return;
				}
		}

		newState[{x, y}] = calculateUAtPos2D({currentState[{x, y}], currentState[{x+1, y}], currentState[{x-1, y}], currentState[{x, y+1}], currentState[{x, y-1}]}, previousState[{x, y}], partitionSize, chamberDefs[chamberNum].c, dt, chamberDefs[chamberNum].mu);
	});

	//currentState = nextState;
	currentStateNum = nextStateNum;
	});
}

void waveChamber2D::setSinglePoint(sycl::vec<double, 2> point, double val){
	array2DWrapper<double> state(state_dats[currentStateNum].data(), state_dats[currentStateNum].size(), partitions);
	auto convertedCoords = point / partitionSize;
	unsigned int index = state.computeIndex((unsigned int)convertedCoords.x(), (unsigned int)convertedCoords.y());
	q.single_task<class setSinglePoind_2d>([=](){
		state.data[index] = val;
	});
}

void waveChamber2D::setPointVarryingTimeFunction(sycl::vec<double, 2> point, std::function<double(double)> valWRTTimeGenerator){
	varryingTimeFuncs.push_back({point, valWRTTimeGenerator});
}

void waveChamber2D::runSimulation(double time, double imageSaveInterval, double printRuntimeStatisticsInterval, double saveRawDataInterval){
	std::cout<<"Simulating for "<<time<<" seconds (dt = "<<dt<<", partition size = "<<partitionSize<<" execution device = ";
	if(q.get_device().is_cpu())
		std::cout<<"[CPU] ";
	if(q.get_device().is_gpu())
		std::cout<<"[GPU] ";
	std::cout<<q.get_device().get_info<hipsycl::sycl::info::device::name>();
	std::cout<<").  Chambers:"<<std::endl;
	std::vector<chamberDef<2>> cpuChambers;
	cpuChambers = chamberDefs.copy_to_host();
	for(unsigned int i=0;i<cpuChambers.size();i++){
		const auto& ch = cpuChambers[i];
		std::cout<<"\tChamber "<<i<<": Pos = {"<<ch.pos.x()<<", "<<ch.pos.y()<<"} (internally {"<<ch.pos_internal.x()<<", "<<ch.pos_internal.y()<<"}), Size = {"<<ch.size.x()<<", "<<ch.size.y()<<"} (internally {"<<ch.size_internal.x()<<", "<<ch.size_internal.y()<<"}), Wave speed c = "<<ch.c<<", Damping μ = "<<ch.mu<<std::endl;
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
	imgWriter.launchThreads(imgWriterThreads);

	if(saveRawDataInterval > 0){
		rawWriter.createFile("rawOutput.grid2dD", partitions);
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

