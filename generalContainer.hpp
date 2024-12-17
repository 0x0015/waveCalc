#pragma once
#include <vector>

template<class T> class generalContainer{
public:
	virtual T* data() = 0;
	virtual unsigned int size() = 0;
	virtual T getElement(unsigned int index) const = 0;
	virtual void setElement(unsigned int index, const T& value) = 0;
};

template<class T> class cpuContainer : public generalContainer<T>{
public:
	std::vector<T> cpuData;
	T* data() override{
		return cpuData.data();
	}
	unsigned int size() override{
		return cpuData.size();
	}
	T getElement(unsigned int index) const override{
		return cpuData[index];
	}
	void setElement(unsigned int index, const T& value) override{
		cpuData[index] = value;
	}
};

