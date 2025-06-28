#pragma once
#include "multiVec.hpp"
#include <array>
#include <string>
#include <memory>
#include <functional>
#include "executionMode.hpp"
#include "generalContainer.hpp"
#include "imageWriter.hpp"
#include "rawDataWriter.hpp"

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

enum simulationEdgeMode{
	REFLECT,
	VOID /*wave will just go off edge of sim and dissapear*/
};

template<EXECUTION_MODE exMode> class waveChamber{
public:
	struct stateVals{
		array2DWrapper<double> uVals;
	};
	vec2<double> size;
	vec2<unsigned int> partitions;
	double partitionSize;
	double dt; //delta time
	using chamberDef = chamberDef;
	generalContainer<exMode, chamberDef> chamberDefs;
	simulationEdgeMode edgeMode;
	std::array<stateVals, 3> states;
	std::array<generalContainer<exMode, double>, 3> state_dats;
	unsigned int currentStateNum = 0;
	stateVals* currentState = &states[currentStateNum];
	imageWriter imgWriter;
	void init(vec2<double> size, double dt, unsigned int xPartitions, std::span<const chamberDef> chambers, simulationEdgeMode edgeMode = REFLECT); //number of partitions for y will be automatially calculated to maintain square partitions
	void step();
	void printuVals();
	void writeRawData();
	void writeToImage(const std::string& filename, double expectedMax);
	void setSinglePoint(vec2<double> point, double val);
	std::vector<std::pair<vec2<double>, std::function<double(double)>>> varryingTimeFuncs;
	void setPointVarryingTimeFunction(vec2<double> point, std::function<double(double)> valWRTTimeGenerator);
	void runSimulation(double time, double imageSaveInterval = -1, double printRuntimeStatisticsInterval = -1, double saveRawDataInterval = -1);
private:
	rawDataWriter rawWriter;
	void initChambers(std::span<const chamberDef> chambers);
	void initStateDats();
	void calculateBestGpuOccupancy();
	int gpuBlockSize, gpuGridSize, gpuMinGridSize;
};

