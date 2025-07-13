#pragma once
#include <thread>
#include <list>
#include <vector>
#include <sycl/sycl.hpp>

template<typename float_t> class imageWriter{
public:
	struct imageWriteRequest{
		std::vector<float_t> data;
		sycl::vec<unsigned int, 2> size;
		std::string filename;
		float_t expectedMax;
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
