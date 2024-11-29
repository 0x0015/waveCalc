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
	std::mutex activeRequestsMutex;
	std::thread processorThread;
	bool threadRunning = false;
	void launchThread();
	void joinThread();
	void createRequest(const imageWriteRequest& request);
	void processRequest(const imageWriteRequest& request);
	void processAllRequestsSynchronous();
	~imageWriter();
private:
	void threadMain();
};
