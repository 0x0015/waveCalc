#pragma once
#include <vector>
#if !defined(DISABLE_GPU_EXECUTION)
#include "gpuLib/deviceVector.hpp"
#endif
#include <span>
#include "executionMode.hpp"

template<EXECUTION_MODE exMode, typename T> struct generalContainer_t{
	using type = struct{};
};
template<typename T> struct generalContainer_t<EXECUTION_MODE_CPU, T>{
	using type = std::vector<T>;
};
#if !defined(DISABLE_GPU_EXECUTION)
template<typename T> struct generalContainer_t<EXECUTION_MODE_GPU, T>{
	using type = agpuUtil::device_vector<T>;
};
#endif

template<EXECUTION_MODE exMode, typename T> using generalContainer = generalContainer_t<exMode, T>::type;

