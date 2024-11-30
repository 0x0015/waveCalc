#pragma once
#include <thread>
#include <list>
#include <vector>
#include "linalg.hpp"

class imageWriter{
public:
	struct imageWriteRequest{
		std::vector<double> data;
		vec2<unsigned int> size;
		std::string filename;
		double expectedMax;
	};
	std::list<imageWriteRequest> activeRequests;
	std::atomic<unsigned int> numActiveRequests;
	std::mutex activeRequestsMutex;
	std::vector<std::thread> processorThreads;
	std::atomic<unsigned int> threadsRunning = 0;
	bool threadsRun = false;
	void launchThreads(unsigned int numThreads);
	void joinThread();
	void createRequest(const imageWriteRequest& request);
	void processRequest(const imageWriteRequest& request);
	void processAllRequestsSynchronous();
	~imageWriter();
private:
	void threadMain();
};
