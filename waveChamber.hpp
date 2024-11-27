#pragma once
#include "multiVec.hpp"
#include <array>
#include <string>
#include <memory>
#include "generalContainer.hpp"

class waveChamber{
public:
	struct stateVals{
		array2DWrapper<double> uVals;
	};
	enum EXECUTION_MODE{
		EXECUTION_MODE_CPU,
		EXECUTION_MODE_GPU
	};
	EXECUTION_MODE executionMode;
	vec2<double> size;
	vec2<unsigned int> partitions;
	double partitionSize;
	double dt; //delta time
	double c; //wave speed
	double mu; //damping constant
	std::array<stateVals, 3> states;
	std::array<std::shared_ptr<generalContainer<double>>, 3> state_dats;
	unsigned int currentStateNum = 0;
	stateVals* currentState = &states[currentStateNum];
	void init(vec2<double> size, double dt, double c, double mu, unsigned int xPartitions, EXECUTION_MODE mode = EXECUTION_MODE_CPU); //number of partitions for y will be automatially calculated
	void step();
	void printuVals();
	void writeToImage(const std::string& filename, double expectedMax);
	void setSinglePoint(vec2<unsigned int> point, double val);
private:
	void initStateDats_cpu();
	void initStateDats_gpu();
	void printuVals_cpu();
	void printuVals_gpu();
	void writeToImage_cpu(const std::string& filename, double expectedMax);
	void writeToImage_gpu(const std::string& filename, double expectedMax);
	void writeToImage_internals(const std::string& filename, double expectedMax, array2DWrapper<double> arrWr);
	void step_cpu();
	void step_gpu();
	void setSinglePoint_cpu(vec2<unsigned int> point, double val);
	void setSinglePoint_gpu(vec2<unsigned int> point, double val);
};
