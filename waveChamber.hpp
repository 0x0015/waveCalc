#pragma once
#include "multiVec.hpp"
#include <array>
#include <string>

struct waveChamber{
	struct stateVals{
		vector2D<double> uVals;
	};
	glm::dvec2 size;
	glm::uvec2 partitions;
	double partitionSize;
	double dt; //delta time
	double c; //wave speed
	double mu; //damping constant
	std::array<stateVals, 3> states;
	unsigned int currentStateNum = 0;
	stateVals* currentState = &states[currentStateNum];
	void init(glm::dvec2 size, double dt, double c, double mu, unsigned int xPartitions); //number of partitions for y will be automatially calculated
	void step();
	void printuVals();
	void writeToImage(const std::string& filename, double expectedMax);
};
