#include "waveChamber.hpp"
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void waveChamber::init(glm::dvec2 s, double _dt, double _c, double _mu, unsigned int xPartitions){
	dt = _dt;
	c = _c;
	mu = _mu;
	size = s;
	partitionSize = s.x / (double)xPartitions;
	double yPartitions = s.y / partitionSize;
	partitions = {xPartitions, (unsigned int)std::round(yPartitions)};
	for(auto& state : states){
		state.uVals = vector2D<double>(partitions);
	}
	currentStateNum = 0;
	currentState = &states[currentStateNum];
}


//based on https://www.csun.edu/~jb715473/math592c/wave2d.pdf

void waveChamber::step(){
	unsigned int nextStateNum = (currentStateNum + 1) % 3;
	unsigned int previousStateNum = (currentStateNum + 2) % 3;
	auto nextState = &states[nextStateNum];
	auto previousState = &states[previousStateNum];

	//set edges to 0
	for(auto i : {0u, partitions.x-1}){
		for(unsigned int j=0;j<partitions.y;j++){
			nextState->uVals[{i, j}] = 0;
		}
	}
	for(auto j : {0u, partitions.y-1}){
		for(unsigned int i=0;i<partitions.x;i++){
			nextState->uVals[{i, j}] = 0;
		}
	}

	const auto calcU = [&](double x, double y){
		double term1 = currentState->uVals[{x, y}] * (4.0-8.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
		double term2 = previousState->uVals[{x, y}] * (mu*dt-2.0) / (mu*dt+2.0);
		double term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
		double term3_pt2 = currentState->uVals[{x+1,y}] + currentState->uVals[{x-1,y}] + currentState->uVals[{x,y+1}] + currentState->uVals[{x, y-1}];
		double term3 = term3_pt1 * term3_pt2;
		return term1 + term2 + term3;
	};

	for(unsigned int i=1;i<partitions.x-1;i++){
		for(unsigned int j=1;j<partitions.y-1;j++){
			nextState->uVals[{i, j}] = calcU(i, j);
		}
	}

	currentState = nextState;
}

void waveChamber::printuVals(){
	for(unsigned int i=0;i<currentState->uVals.size.x;i++){
		for(unsigned int j=0;j<currentState->uVals.size.y;j++){
			std::cout<<currentState->uVals[{i, j}]<<" ";
		}
		std::cout<<std::endl;
	}
}

void waveChamber::writeToImage(const std::string& filename, double expectedMax){
	std::vector<uint8_t> imageData(partitions.x * partitions.y*3);

	unsigned int x=0;
	unsigned int y=0;
	for(unsigned int i=0;i<imageData.size();i+=3){
		double val = currentState->uVals[{x, y}];
		double scaled = (val) / expectedMax * 255;
		imageData[i+1] = 0;
		if(scaled < 0){
			imageData[i] = -scaled;
			imageData[i+2] = 0;
		}else{
			imageData[i] = 0;
			imageData[i+2] = scaled;
		}
		x++;
		if(x >= partitions.x){
			x = 0;
			y++;
		}
	}

	stbi_write_png(filename.c_str(), partitions.x, partitions.y, 3, imageData.data(), partitions.x * 3);
}

