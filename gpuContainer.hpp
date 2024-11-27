#pragma once
#include "generalContainer.hpp"
#include "gpuLib/deviceVector.hpp"

template<class T> class gpuContainer : public generalContainer<T>{
public:
	hipUtil::device_vector<T> gpuData;
	T* data() override{
		return gpuData.data();
	}
	unsigned int size() override{
		return gpuData.size();
	}
};

