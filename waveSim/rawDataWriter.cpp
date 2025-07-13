#include "rawDataWriter.hpp"
#include <optional>
#include "multiVec.hpp"
#include <iostream>

using namespace std::chrono_literals;

template<typename float_t> void rawDataWriter<float_t>::threadMain(){
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

template<typename float_t> void rawDataWriter<float_t>::createFile(const std::string& filename, sycl::vec<unsigned int, 2> gridSize, float_t sampleRate){
	file.open(filename, std::ios_base::out | std::ios_base::binary);
	size = gridSize;
	constexpr uint32_t fileVersionNum = 1;
	constexpr unsigned int extraPadding = 32;
	file << "2dDGridFile";
	file.write((char*)&fileVersionNum, sizeof(fileVersionNum));
	file.write((char*)&size.x(), sizeof(size.x()));
	file.write((char*)&size.y(), sizeof(size.y()));
	file.write((char*)&sampleRate, sizeof(sampleRate));
	for(unsigned int i=0;i<extraPadding;i++)
		file << (uint8_t)0;
}

template<typename float_t> void rawDataWriter<float_t>::launchThread(){
	threadRunning = true;
	processorThread = std::thread([&](){threadMain();});
}

template<typename float_t> void rawDataWriter<float_t>::joinThreadAndClose(){
	threadRunning = false;
	if(processorThread.joinable())
		processorThread.join();
	processAllRequestsSynchronous();
	if(file.is_open())
		file.close();
}

template<typename float_t> rawDataWriter<float_t>::~rawDataWriter(){
	threadRunning = false;
	if(processorThread.joinable())
		processorThread.join();
	if(file.is_open())
		file.close();
}

template<typename float_t> void rawDataWriter<float_t>::createRequest(const rawWriteRequest& request){
	activeRequestsMutex.lock();
	activeRequests.push_back(request);
	activeRequestsMutex.unlock();
	numActiveRequests++;
}

template<typename float_t> void rawDataWriter<float_t>::processAllRequestsSynchronous(){
	for(const auto& request : activeRequests){
		processRequest(request);
	}
	activeRequests.clear();
	numActiveRequests = 0;
}

template<typename float_t> void rawDataWriter<float_t>::processRequest(const rawWriteRequest& request){
	file.write((char*)request.data.data(), request.data.size() * sizeof(float_t));
}

template class rawDataWriter<float>;
template class rawDataWriter<double>;

