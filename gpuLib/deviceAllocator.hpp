#pragma once
#include "hipUtil.hpp"

namespace hipUtil{

template<class T> class deviceAllocator{
public:
	using value_type = T;
	deviceAllocator() = default;
	template<class U> deviceAllocator(deviceAllocator<U> const&) noexcept{}

	[[nodiscard]] value_type* allocate(std::size_t n){
		value_type* output;
		check_error(agpuMalloc(&output, n * sizeof(T)));
		return output;
	}
	void deallocate(T* p, std::size_t n) noexcept{
		auto error = agpuFree(p);
		if(error != agpuSuccess){
			//do nothing right now, as we want noexcept
		}
	}
};

template<class T, class U> bool operator==(const deviceAllocator <T>&, const deviceAllocator <U>&) { return true; } 
template<class T, class U> bool operator!=(const deviceAllocator <T>&, const deviceAllocator <U>&) { return false; }

template<typename T> struct is_device : std::false_type{};
template<template<typename, typename> typename Outer, typename Inner> struct is_device<Outer<Inner, deviceAllocator<Inner>>> : std::true_type{};
template<typename T> constexpr static auto is_unified_v = is_device<T>::value;

}
