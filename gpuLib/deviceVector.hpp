#pragma once
#include "deviceAllocator.hpp"
#include <vector>
#include <span>

namespace agpuUtil{

namespace impl{
	template<class T> __global__ void better_memset(T* mem, T val){	
		const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
		mem[gid] = val;
	}
}

template<class T> class device_vector{
public:
	using value_type = T;
	device_vector(){
		ptr = nullptr;
		internal_size = 0;
		internal_capacity = 0;
	}
	device_vector(size_t default_size){
		check_error(agpuMalloc(&ptr, sizeof(T) * default_size));
		internal_capacity = default_size;
		internal_size = default_size;
	}
	device_vector(size_t default_size, const T& v){
		check_error(agpuMalloc(&ptr, sizeof(T) * default_size));
		/*
		if constexpr(sizeof(T) == sizeof(uint8_t)){
			check_error(hipMemsetD8(ptr, v, sizeof(T) * default_size));
		}else if constexpr(sizeof(T) == sizeof(uint16_t)){
			check_error(hipMemsetD16(ptr, v, sizeof(T) * default_size));
		}else if constexpr(sizeof(T) == sizeof(uint32_t)){
			check_error(hipMemsetD32(ptr, v, sizeof(T) * default_size));
		}else if constexpr(sizeof(T) == sizeof(uint64_t)){
			check_error(hipMemsetD64(ptr, v, sizeof(T) * default_size));
		*/
		//only works when using hipMemAllocPitch()
		/*
		if constexpr(sizeof(T) == sizeof(uint8_t)){
			check_error(hipMemset(ptr, v, sizeof(T) * default_size));
		}else{
			std::vector<T> temp(default_size, v);
			check_error(hipMemcpy(ptr, temp.data(), sizeof(T) * default_size, hipMemcpyHostToDevice));
		}
		*/

		constexpr unsigned int blockSize = 256;
		impl::better_memset<T><<<default_size/blockSize+ (default_size % blockSize != 0), blockSize>>>(ptr, v);

		check_error(agpuDeviceSynchronize());

		internal_capacity = default_size;
		internal_size = default_size;
	}
	device_vector(const std::initializer_list<T>& i_list){
		int sz = i_list.size();

		check_error(agpuMalloc(&ptr, sizeof(T) * sz));

		check_error(agpuMemcpy(ptr, std::data(i_list), sizeof(T) * sz, agpuMemcpyHostToDevice));

		internal_size = sz;
		internal_capacity = sz;
	}
	device_vector(std::span<const T>& vec){
		int sz = vec.size();

		check_error(agpuMalloc(&ptr, sizeof(T) * sz));
		check_error(agpuMemcpy(ptr, vec.data(), sizeof(T) * sz, agpuMemcpyHostToDevice));

		internal_size = sz;
		internal_capacity = sz;
	}
	device_vector(const std::vector<T>& vec){
		int sz = vec.size();

		check_error(agpuMalloc(&ptr, sizeof(T) * sz));
		check_error(agpuMemcpy(ptr, vec.data(), sizeof(T) * sz, agpuMemcpyHostToDevice));

		internal_size = sz;
		internal_capacity = sz;
	}
	template<size_t S> device_vector(const std::array<T, S>& vec){
		constexpr size_t sz = S;

		check_error(agpuMalloc(&ptr, sizeof(T) * sz));
		check_error(agpuMemcpy(ptr, vec.data(), sizeof(T) * sz, agpuMemcpyHostToDevice));

		internal_size = sz;
		internal_capacity = sz;
	}
	device_vector(const device_vector<T>& vec){	
		int sz = vec.size();

		check_error(agpuMalloc(&ptr, sizeof(T) * sz));
		check_error(agpuMemcpy(ptr, vec.data(), sizeof(T) * sz, agpuMemcpyDeviceToDevice));

		internal_size = sz;
		internal_capacity = sz;
	}
	template<typename... ARGS> void emplace_back(ARGS&&... args){
		if(internal_size == internal_capacity){
			if(ptr == nullptr){
				internal_capacity = 1;
				internal_size = 1;
				reallocate();
				ptr[0] = std::move(T(std::forward<ARGS>(args)...));
			}else if(internal_size < 8){
				internal_capacity++;
				reallocate();
				ptr[internal_size] = std::move(T(std::forward<ARGS>(args)...));
				internal_size++;
			}else{
				internal_capacity*=2;
				reallocate();
				ptr[internal_size] = std::move(T(std::forward<ARGS>(args)...));
				internal_size++;
			}
		}
	}
	void push_back(const T& v){
		emplace_back(v);
	}
	void push_back(T&& v){
		emplace_back(v);
	}
	void pop_back(){
		internal_size--;
	}
	void clear(){
		internal_size = 0;
	}
	void shrink_to_fit(){
		internal_capacity = internal_size;
		reallocate();
	}
	void reserve(size_t n){
		if(n > internal_capacity){
			internal_capacity = n;
			reallocate();
		}
	}
	void resize(size_t n){
		if(n > internal_capacity){
			internal_capacity = n;
			reallocate();
		}
		internal_size = n;
	}
	std::vector<T> copy_to_host(){
		std::vector<T> output(internal_size);
		check_error(agpuMemcpy(output.data(), ptr, internal_size * sizeof(T), agpuMemcpyDeviceToHost));
		return output;
	}
	constexpr size_t size() const noexcept{
		return internal_size;
	}
	constexpr size_t capacity() const noexcept{
		return internal_capacity;
	}
	constexpr T* data() noexcept{
		return ptr;
	}
	constexpr const T* data() const noexcept{
		return ptr;
	}
	constexpr T& operator[](size_t pos){
		return  ptr[pos];
	}
	constexpr const T& operator[](size_t pos) const{
		return  ptr[pos];
	}
	device_vector& operator=(const device_vector& other){
		if(this == &other)
			return *this;

		if(other.size() > internal_capacity){
			internal_capacity = other.size();
			reallocate();
		}

		check_error(agpuMemcpy(ptr, other.data(), other.size() * sizeof(T), agpuMemcpyDeviceToDevice));
		internal_size = other.size();

		return *this;
	}
	device_vector& operator=(const std::vector<T>& other){

		if(other.size() > internal_capacity){
			internal_capacity = other.size();
			reallocate();
		}

		check_error(agpuMemcpy(ptr, other.data(), other.size() * sizeof(T), agpuMemcpyHostToDevice));
		internal_size = other.size();

		return *this;
	}
	operator T*(){
		return ptr;
	}
	operator const T*() const{
		return ptr;
	}
	~device_vector(){
		if(ptr == nullptr)
			return;
		check_error(agpuDeviceSynchronize());
		check_error(agpuFree(ptr));
	}
private:
	T* ptr;
	size_t internal_size;
	size_t internal_capacity;
	void reallocate(){
		T* temp;
		check_error(agpuDeviceSynchronize());
		check_error(agpuMalloc(&temp, internal_capacity * sizeof(T)));
		if(ptr != nullptr){
			check_error(agpuMemcpy(temp, ptr, internal_size * sizeof(T), agpuMemcpyDeviceToDevice));
		}

		check_error(agpuFree(ptr));
		ptr = temp;
	}
};

template<class T> struct is_device<device_vector<T>> : std::true_type{};

}
