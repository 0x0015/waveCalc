#pragma once
#include "linalg.hpp"
#include "gpuLib/agpuCallableMember.hpp"

template<typename T> class array2DWrapper{
public:
	vec2<unsigned int> size;
	T* data;
	unsigned int length;
	AGPU_CALLABLE_MEMBER array2DWrapper() = default;
	AGPU_CALLABLE_MEMBER array2DWrapper(T* d, unsigned int l, vec2<unsigned int> s) : size(s), data(d), length(l){
	}
	AGPU_CALLABLE_MEMBER array2DWrapper(T* d, unsigned int l, unsigned int width, unsigned int height) : array2DWrapper(d, l, vec2<unsigned int>{width, height}){}
	AGPU_CALLABLE_MEMBER T& get(unsigned int x, unsigned int y){
		return data[size.x * y + x];
	}
	AGPU_CALLABLE_MEMBER const T& get(unsigned int x, unsigned int y) const{
		return data[size.x * y + x];
	}
	AGPU_CALLABLE_MEMBER bool operator==(const array2DWrapper& other){
		return data == other.data;
	}
	AGPU_CALLABLE_MEMBER T& operator[](vec2<unsigned int> s){
		return get(s.x, s.y);
	}
	AGPU_CALLABLE_MEMBER const T& operator[](vec2<unsigned int> s) const{
		return get(s.x, s.y);
	}
	template<class func> AGPU_CALLABLE_MEMBER void foreach(const func& f){
		for(unsigned int i=0;i<size.x;i++){
			for(unsigned int j=0;j<size.y;j++){
				f(get(i, j), i, j);
			}
		}
	}
};

template<typename T> class array2DWrapper_const{
public:
	vec2<unsigned int> size;
	const T* data;
	unsigned int length;
	AGPU_CALLABLE_MEMBER array2DWrapper_const() = default;
	AGPU_CALLABLE_MEMBER array2DWrapper_const(const T* d, unsigned int l, vec2<unsigned int> s) : size(s), data(d), length(l){
	}
	AGPU_CALLABLE_MEMBER array2DWrapper_const(const T* d, unsigned int l, unsigned int width, unsigned int height) : array2DWrapper_const(d, l, vec2<unsigned int>{width, height}){}
	AGPU_CALLABLE_MEMBER const T& get(unsigned int x, unsigned int y) const{
		return data[size.x * y + x];
	}
	AGPU_CALLABLE_MEMBER bool operator==(const array2DWrapper_const& other){
		return data == other.data;
	}
	AGPU_CALLABLE_MEMBER const T& operator[](vec2<unsigned int> s) const{
		return get(s.x, s.y);
	}
	template<class func> AGPU_CALLABLE_MEMBER void foreach(const func& f){
		for(unsigned int i=0;i<size.x;i++){
			for(unsigned int j=0;j<size.y;j++){
				f(get(i, j), i, j);
			}
		}
	}
};

