#pragma once

#define AGPU_INCLUDED
#define AGPU_DEVICE_PREFIX __device__
#define AGPU_GLOBAL_PREFIX __global__

#if AGPU_BACKEND_HIP
#define AGPU_BACKEND_FOUND
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
#define agpuOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

#endif

#ifdef AGPU_BACKEND_CUDA
#define AGPU_BACKEND_FOUND
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
#define agpuOccupancyMaxPotentialBlockSize cudaOccupancyMaxPotentialBlockSize

#endif

#ifndef AGPU_BACKEND_FOUND
//if no gpu backend is found, just stub it all so compilers don't complain
#include <cstdint>
using agpu_half = int16_t;
using agpu_bfloat16 = int16_t;
using agpuError_t = int;
constexpr inline agpuError_t agpuSuccess = 1;
constexpr inline int agpuMemcpyHostToDevice = 1;
constexpr inline int agpuMemcpyDeviceToHost = 2;
constexpr inline int agpuMemcpyDeviceToDevice = 3;
struct agpuDeviceProp_t{};

constexpr agpuError_t agpuGetDevice(int* device){*device = -1;return -1;}
constexpr agpuError_t agpuFree(void* devPtr){return -1;}
constexpr const char* agpuGetErrorString(agpuError_t err){return "No AGPU backend found";}
constexpr agpuError_t agpuMalloc(void** devPtr, std::size_t size){return -1;}
constexpr agpuError_t agpuDeviceSynchronize(){return -1;}
constexpr agpuError_t agpuMemcpy(void* dst, const void* src, std::size_t count, int memcpyKind){return -1;}
constexpr agpuError_t agpuGetDeviceProperties(agpuDeviceProp_t* prop, int device){return -1;}
template<typename T> agpuError_t agpuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func, std::size_t dynamicSMemSize = 0, int blockSizeLimit = 0){return -1;}

#define __global__ 
constexpr inline struct{unsigned int x; unsigned int y; unsigned int z;} blockIdx{};
constexpr inline struct{unsigned int x; unsigned int y; unsigned int z;} blockDim{};
constexpr inline struct{unsigned int x; unsigned int y; unsigned int z;} threadIdx{};

#define KERNEL_LAUNCH(kernel, gridDim, blockDim) if(true) kernel

#else
#define KERNEL_LAUNCH(kernel, gridDim, blockDim) kernel<<<gridDim, blockDim>>>
#endif

