#pragma once
#include "generalContainer.hpp"
#include "gpuLib/deviceVector.hpp"

namespace gpuContainer_impl{
	template<class T> __global__ void setElement(T* array, unsigned int index, T value){
		const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
		if(gid == 0)
			array[index] = value;
	}
	template<class T> __global__ void getElement(T* array, unsigned int index, T* output){
		const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
		if(gid == 0)
			*output = array[index];
	}
}

template<class T> class gpuContainer : public generalContainer<T>{
public:
	hipUtil::device_vector<T> gpuData;
	T* data() override{
		return gpuData.data();
	}
	unsigned int size() override{
		return gpuData.size();
	}
	T getElement(unsigned int index) const override{
		hipUtil::device_vector<T> output;
		output.resize(1);
		gpuContainer_impl::getElement<T><<<1, 1>>>(gpuData.data(), index, output.data());
		return output.copy_to_host()[0];
	}
	void setElement(unsigned int index, const T& value) override{
		gpuContainer_impl::setElement<T><<<1, 1>>>(gpuData.data(), index, value);
	}
};

