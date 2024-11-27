#pragma once

#ifndef AGPU_BACKEND
	#ifdef hipGetDeviceProperties
		#define AGPU_BACKEND HIP
	#else
		#define AGPU_BACKEND CUDA
	#endif
#endif

#if AGPU_BACKEND == HIP
#include <hip/hip_runtime.h>

#endif

#if AGPU_BACKEND == CUDA

#endif

