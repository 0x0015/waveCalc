#pragma once
#include "multiVec.hpp"
#include <array>
#include <string>
#include <memory>
#include "generalContainer.hpp"
#include "imageWriter.hpp"
#include "rawDataWriter.hpp"
#include "waveChamber.hpp"

class waveSimulator{
public:
	struct stateVals{
		array2DWrapper<double> uVals;
	};
	waveEnvironment environment;
	imageWriter imgWriter;
	double dt;
	void init(const waveEnvironment& waveEnv, double dt); //number of partitions for y will be automatially calculated
	void step();
	void printuVals();
	void writeRawData();
	void writeToImage(const std::string& filename, double expectedMax);
	void setSinglePoint(vec2<unsigned int> point, double val);
	void runSimulation(double time, double imageSaveInterval = -1, double printRuntimeStatisticsInterval = -1, double saveRawDataInterval = -1);
private:
	waveEnvironmentAccessor cpuAccessor;
	rawDataWriter rawWriter;
	void initStateDats_cpu();
	void initStateDats_gpu();
	void printuVals_cpu();
	void printuVals_gpu();
	void writeRawData_cpu();
	void writeRawData_gpu();
	void writeToImage_cpu(const std::string& filename, double expectedMax);
	void writeToImage_gpu(const std::string& filename, double expectedMax);
	void step_cpu();
	void step_gpu();
	void setSinglePoint_cpu(vec2<unsigned int> point, double val);
	void setSinglePoint_gpu(vec2<unsigned int> point, double val);
	void calculateBestGpuOccupancy();
	int gpuBlockSize, gpuGridSize, gpuMinGridSize;
};
