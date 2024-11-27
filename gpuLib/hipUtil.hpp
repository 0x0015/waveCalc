//MIT LICENSE UNDER 0x15
//USE AT YOUR OWN RISK (this was written in one afternoon)

#pragma once
#include <stdexcept>
#include "agpuInterface.hpp"

namespace hipUtil{

void check_error(const hipError_t error){
	if(error != hipSuccess){
		throw std::runtime_error{hipGetErrorString(error)};
	}
};

int get_current_device(){
	int result;
	check_error(hipGetDevice(&result));
	return result;
}

}

