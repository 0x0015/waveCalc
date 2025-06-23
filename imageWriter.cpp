#include "imageWriter.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <optional>
#include "multiVec.hpp"
#include <iostream>

using namespace std::chrono_literals;

void imageWriter::threadMain(){
	threadsRunning++;
	while(threadsRun){
		std::optional<imageWriteRequest> currentRequest = std::nullopt;
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
	threadsRunning--;
}

void imageWriter::launchThreads(unsigned int numThreads){
#ifdef IMAGE_WRITER_DEBUG
	std::cout<<"Launching "<<numThreads<<" imageWriter threads"<<std::endl;
#endif
	threadsRun = true;
	for(unsigned int i=0;i<numThreads;i++){
		processorThreads.push_back(std::thread([&](){threadMain();}));
	}
}

void imageWriter::joinThread(){
#ifdef IMAGE_WRITER_DEBUG
	std::cout<<"Joining imageWriter thread"<<std::endl;
#endif
	threadsRun = false;
	for(auto& thread : processorThreads){
		if(thread.joinable())
			thread.join();
	}
	if(threadsRunning > 0){
		std::cout<<"Warning: when joining imageWriter threads "<<threadsRunning<<" threads did not stop properly (either still running or finished prematurely)"<<std::endl;
	}
	processAllRequestsSynchronous();
}

imageWriter::~imageWriter(){
	threadsRun = false;
	for(auto& thread : processorThreads){
		if(thread.joinable())
			thread.join();
	}
}

void imageWriter::createRequest(const imageWriteRequest& request){
	activeRequestsMutex.lock();
	activeRequests.push_back(request);
	activeRequestsMutex.unlock();
	numActiveRequests++;
}

void imageWriter::processAllRequestsSynchronous(){
	for(const auto& request : activeRequests){
		processRequest(request);
	}
	activeRequests.clear();
	numActiveRequests = 0;
}

void imageWriter::processRequest(const imageWriteRequest& request){
#ifdef IMAGE_WRITER_DEBUG
	std::cout<<"Processing imageWriter request"<<std::endl;
#endif
	std::vector<uint8_t> imageData(request.size.x * request.size.y*3);
	auto arrWr = array2DWrapper_view<double>(request.data.data(), request.data.size(), request.size);

	unsigned int x=0;
	unsigned int y=0;
	for(unsigned int i=0;i<imageData.size();i+=3){
		double val = arrWr[{x, y}];
		double scaled = (val) / request.expectedMax * 255;
		imageData[i+1] = 0;
		if(scaled < 0){
			imageData[i] = -scaled;
			imageData[i+2] = 0;
		}else{
			imageData[i] = 0;
			imageData[i+2] = scaled;
		}
		x++;
		if(x >= request.size.x){
			x = 0;
			y++;
		}
	}

	stbi_write_png(request.filename.c_str(), request.size.x, request.size.y, 3, imageData.data(), request.size.x * 3);
	numActiveRequests--;
}
