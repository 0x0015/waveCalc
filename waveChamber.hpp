#pragma once
#include "multiVec.hpp"
#include <array>
#include <string>
#include <memory>
#include "executionMode.hpp"
#include "generalContainer.hpp"
#include "imageWriter.hpp"
#include "rawDataWriter.hpp"

template<EXECUTION_MODE exMode> class waveChamber{
public:
	struct stateVals{
		array2DWrapper<double> uVals;
	};
	vec2<double> size;
	vec2<unsigned int> partitions;
	double partitionSize;
	double dt; //delta time
	struct chamberDef{
		vec2<double> pos;
		vec2<double> size;
		double c; //wave speed
		double mu; //damping constant
		vec2<unsigned int> pos_internal;
		vec2<unsigned int> size_internal;
		AGPU_CALLABLE_MEMBER inline constexpr bool isPointInChamber(vec2<unsigned int> point) const{
			return(point.x >= pos_internal.x && point.y >= pos_internal.y && point.x < pos_internal.x + size_internal.x && point.y < pos_internal.y + size_internal.y);
		}
	};
	generalContainer<exMode, chamberDef> chamberDefs;
	std::array<stateVals, 3> states;
	std::array<generalContainer<exMode, double>, 3> state_dats;
	unsigned int currentStateNum = 0;
	stateVals* currentState = &states[currentStateNum];
	imageWriter imgWriter;
	void init(vec2<double> size, double dt, unsigned int xPartitions, std::span<const chamberDef> chambers); //number of partitions for y will be automatially calculated
	void step();
	void printuVals();
	void writeRawData();
	void writeToImage(const std::string& filename, double expectedMax);
	void setSinglePoint(vec2<unsigned int> point, double val);
	void runSimulation(double time, double imageSaveInterval = -1, double printRuntimeStatisticsInterval = -1, double saveRawDataInterval = -1);
private:
	rawDataWriter rawWriter;
	void initChambers(std::span<const chamberDef> chambers);
	void initStateDats();
	void calculateBestGpuOccupancy();
	int gpuBlockSize, gpuGridSize, gpuMinGridSize;
};

