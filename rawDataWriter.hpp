#pragma once
#include <thread>
#include <list>
#include <vector>
#include <fstream>
#include <sycl/sycl.hpp>

class rawDataWriter{
public:
	std::string filename;
	std::ofstream file;
	sycl::vec<unsigned int, 2> size;
	struct rawWriteRequest{
		std::vector<double> data;
	};
	std::list<rawWriteRequest> activeRequests;
	std::atomic<unsigned int> numActiveRequests;
	std::mutex activeRequestsMutex;
	std::thread processorThread;
	bool threadRunning = false;
	void createFile(const std::string& filename, sycl::vec<unsigned int, 2> stateSize);
	void launchThread();
	void joinThreadAndClose();
	void createRequest(const rawWriteRequest& request);
	void processRequest(const rawWriteRequest& request);
	void processAllRequestsSynchronous();
	~rawDataWriter();
private:
	void threadMain();
};
