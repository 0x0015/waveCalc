#pragma once
#include "hipUtil.hpp"
#include "deviceVector.hpp"
#include <cstdint>

#ifdef AGPU_BACKEND_HIP
#include <hipcub/hipcub.hpp>

#define DeviceRadixSort_SortKeys hipcub::DeviceRadixSort::SortKeys
#define DeviceReduce_Sum hipcub::DeviceReduce::Sum
#endif

#ifdef AGPU_BACKEND_CUDA
#include <cub/cub.cuh>

#define DeviceRadixSort_SortKeys cub::DeviceRadixSort::SortKeys
#define DeviceReduce_Sum cub::DeviceReduce::Sum
#endif

namespace hipUtil{

namespace impl{
	template<class T> constexpr bool isStandardType(){
		return (std::is_same_v<T, uint8_t> ||
			std::is_same_v<T, uint16_t> ||
			std::is_same_v<T, uint32_t> ||
			std::is_same_v<T, uint64_t> ||
			std::is_same_v<T, int8_t> ||
			std::is_same_v<T, int16_t> ||
			std::is_same_v<T, int32_t> ||
			std::is_same_v<T, int64_t> ||
			std::is_same_v<T, float> ||
			std::is_same_v<T, double> ||
			std::is_same_v<T, agpu_half> ||
			std::is_same_v<T, agpu_bfloat16>);
	}
}

template<class T> typename std::enable_if_t<impl::isStandardType<typename T::value_type>(), void> sort(T& vec){
	using itemType = typename T::value_type;
	device_vector<itemType> out(vec.size());

	size_t temp_storage_bytes = 0;
	check_error(DeviceRadixSort_SortKeys(nullptr, temp_storage_bytes, vec.data(), out.data(), vec.size()));
	device_vector<uint8_t> temp(temp_storage_bytes);
	check_error(DeviceRadixSort_SortKeys(temp.data(), temp_storage_bytes, vec.data(), out.data(), vec.size()));

	//no synchronize needed as memcopy wil force one
	vec = out;
}


//although this should be supported (see https://github.com/ROCm/hipCUB/blob/3e1780a8ab573253f6a20ac28c9c6fe2d1571815/hipcub/include/hipcub/backend/rocprim/device/device_radix_sort.hpp#L508), it doesn't seem to be packaged in hipcub 6.0.2-1 on arch
/*
template<class T, class Decomposer> void sort(T& vec, Decomposer dec){
	using itemType = typename T::value_type;
	device_vector<itemType> out(vec.size());

	size_t temp_storage_bytes = 0;
	check_error(hipcub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, vec.data(), out.data(), vec.size(), dec));
	device_vector<uint8_t> temp(temp_storage_bytes);
	check_error(hipcub::DeviceRadixSort::SortKeys(temp.data(), temp_storage_bytes, vec.data(), out.data(), vec.size(), dec));

	//no synchronize needed as memcopy wil force one
	vec = out;
}
*/

namespace impl{
	template<class T> __global__ void fill_impl(T* mem, T val){	
		const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
		mem[gid] = val;
	}
}

template<class T> void fill(T& vec, const typename T::value_type& val){
	using itemType = typename T::value_type;
	
	constexpr unsigned int blockSize = 256;
	impl::fill_impl<itemType><<<vec.size()/blockSize + (vec.size() % blockSize != 0), blockSize>>>(vec.data(), val);

	check_error(agpuDeviceSynchronize());
}

namespace impl{
	template<class T, class Generator> __global__ void generate_impl(T* mem, Generator g){	
		const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
		mem[gid] = g();
	}
}

template<class T, class Generator> void generate(T& vec, Generator g){	
	using itemType = typename T::value_type;
	
	constexpr unsigned int blockSize = 256;
	impl::generate_impl<itemType><<<vec.size()/blockSize + (vec.size() % blockSize != 0), blockSize>>>(vec.data(), g);

	check_error(agpuDeviceSynchronize());
}

namespace impl{
	template<class T, class Op> __global__ void transform_impl(T* mem, Op op){	
		const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
		mem[gid] = op(mem[gid]);
	}
}

template<class T, class Op> void transform(T& vec, Op op){
	using itemType = typename T::value_type;

	constexpr unsigned int blockSize = 256;
	impl::transform_impl<itemType><<<vec.size()/blockSize + (vec.size() % blockSize != 0), blockSize>>>(vec.data(), op);

	check_error(agpuDeviceSynchronize());
}


template<class T> typename T::value_type accumulate(const T& vec){
	using itemType = typename T::value_type;

	device_vector<itemType> out(1);
	size_t temp_storage_bytes = 0;
	check_error(DeviceReduce_Sum(nullptr, temp_storage_bytes, vec.data(), out.data(), vec.size()));
	device_vector<uint8_t> temp(temp_storage_bytes);

	check_error(DeviceReduce_Sum(temp.data(), temp_storage_bytes, vec.data(), out.data(), vec.size()));

	check_error(agpuDeviceSynchronize());

	return out[0];
}


}
