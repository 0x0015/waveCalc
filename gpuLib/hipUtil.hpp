//MIT LICENSE UNDER 0x15
//USE AT YOUR OWN RISK (this was written in one afternoon)

#pragma once
#include <stdexcept>
#include "agpuInterface.hpp"

namespace hipUtil{

void check_error(const agpuError_t error){
	if(error != agpuSuccess){
		throw std::runtime_error{std::string("Cuda error: ") + agpuGetErrorString(error)};
	}
};

int get_current_device(){
	int result;
	check_error(agpuGetDevice(&result));
	return result;
}

}

