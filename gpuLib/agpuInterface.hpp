#pragma once

#ifndef AGPU_BACKEND
	#ifdef hipGetDeviceProperties
		#define AGPU_BACKEND_HIP
	#else
		#define AGPU_BACKEND_CUDA
	#endif
#endif

#define AGPU_INCLUDED
#define AGPU_DEVICE_PREFIX __device__
#define AGPU_GLOBAL_PREFIX __global__

#if AGPU_BACKEND_HIP
#include <hip/hip_runtime.h>

#define agpu_bfloat16 hip_bfloat16
#define agpu_half half
#define agpuError_t hipError_t
#define agpuSuccess hipSuccess
#define agpuMemcpyHostToDevice hipMemcpyHostToDevice
#define agpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define agpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define agpuDeviceProp_t hipDeviceProp_t

#define agpuGetDevice hipGetDevice
#define agpuFree hipFree
#define agpuGetErrorString hipGetErrorString
#define agpuMalloc hipMalloc
#define agpuDeviceSynchronize hipDeviceSynchronize
#define agpuMemcpy hipMemcpy
#define agpuGetDeviceProperties hipGetDeviceProperties

#endif

#ifdef AGPU_BACKEND_CUDA
#include <cuda_runtime.h>

#define agpu_half __half
#define agpu_bfloat16 __nv_bfloat16
#define agpuError_t cudaError_t
#define agpuSuccess cudaSuccess
#define agpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define agpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define agpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define agpuDeviceProp_t cudaDeviceProp

#define agpuGetDevice cudaGetDevice
#define agpuFree cudaFree
#define agpuGetErrorString cudaGetErrorString
#define agpuMalloc cudaMalloc
#define agpuDeviceSynchronize cudaDeviceSynchronize
#define agpuMemcpy cudaMemcpy
#define agpuGetDeviceProperties cudaGetDeviceProperties

#endif

