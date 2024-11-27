#pragma once
#include "hipUtil.hpp"

namespace hipUtil{

template<class T> class unifiedAllocator{
public:
	using value_type = T;
	unifiedAllocator() = default;
	template<class U> unifiedAllocator(unifiedAllocator<U> const&) noexcept{}

	[[nodiscard]] value_type* allocate(std::size_t n){
		value_type* output;
		check_error(hipMallocManaged(&output, n * sizeof(T)));
		return output;
	}
	void deallocate(T* p, std::size_t n) noexcept{
		auto error = hipFree(p);
		if(error != hipSuccess){
			//do nothing right now, as we want noexcept
		}
	}
};

template<class T, class U> bool operator==(const unifiedAllocator <T>&, const unifiedAllocator <U>&) { return true; } 
template<class T, class U> bool operator!=(const unifiedAllocator <T>&, const unifiedAllocator <U>&) { return false; }

template<typename T> struct is_unified : std::false_type{};
template<template<typename, typename> typename Outer, typename Inner> struct is_unified<Outer<Inner, unifiedAllocator<Inner>>> : std::true_type{};
template<typename T> constexpr static auto is_unified_v = is_unified<T>::value;

template <typename T, typename = std::enable_if_t<is_unified_v<T>>>
auto prefetch(T const & container,  hipStream_t stream = 0, int device = get_current_device()){
	using value_type = typename T::value_type;
	auto p = container.data();
	if (p) {
		check_error(hipMemPrefetchAsync(p, container.size() * sizeof(value_type), device, stream));
	}
}

}
