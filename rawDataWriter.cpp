#include "rawDataWriter.hpp"
#include <optional>
#include "multiVec.hpp"
#include <iostream>

using namespace std::chrono_literals;

void rawDataWriter::threadMain(){
	while(threadRunning){
		std::optional<rawWriteRequest> currentRequest = std::nullopt;
		if(numActiveRequests > 0){
			activeRequestsMutex.lock();
			if(!activeRequests.empty()){
				currentRequest = activeRequests.front();
				activeRequests.pop_front();
			}
			activeRequestsMutex.unlock();
		}

		if(currentRequest){
			processRequest(*currentRequest);
			currentRequest = std::nullopt;
		}else{
			std::this_thread::sleep_for(0.1s);
		}
	}
	threadRunning = false;
}

void rawDataWriter::createFile(const std::string& filename, vec2<unsigned int> gridSize){
	file.open(filename, std::ios_base::out | std::ios_base::binary);
	size = gridSize;
	constexpr unsigned int fileVersionNum = 1;
	constexpr unsigned int extraPadding = 32;
	file << "2dDGridFile";
	file << fileVersionNum;
	file << size.x;
	file << size.y;
	for(unsigned int i=0;i<extraPadding;i++)
		file << (uint8_t)0;
}

void rawDataWriter::launchThread(){
	threadRunning = true;
	processorThread = std::thread([&](){threadMain();});
}

void rawDataWriter::joinThreadAndClose(){
	threadRunning = false;
	if(processorThread.joinable())
		processorThread.join();
	processAllRequestsSynchronous();
	if(file.is_open())
		file.close();
}

rawDataWriter::~rawDataWriter(){
	threadRunning = false;
	if(processorThread.joinable())
		processorThread.join();
	if(file.is_open())
		file.close();
}

void rawDataWriter::createRequest(const rawWriteRequest& request){
	activeRequestsMutex.lock();
	activeRequests.push_back(request);
	activeRequestsMutex.unlock();
	numActiveRequests++;
}

void rawDataWriter::processAllRequestsSynchronous(){
	for(const auto& request : activeRequests){
		processRequest(request);
	}
	activeRequests.clear();
	numActiveRequests = 0;
}

void rawDataWriter::processRequest(const rawWriteRequest& request){	
	auto arrWr = array2DWrapper_const<double>(request.data.data(), request.data.size(), size);

	for(unsigned int j=0;j<size.y;j++){
		for(unsigned int i=0;i<size.x;i++){
			file << arrWr[{i, j}];
		}
	}
}
