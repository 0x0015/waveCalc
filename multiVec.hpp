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
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(unsigned int x, unsigned int y) const{
		return size.x * y + x;
	}
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(vec2<unsigned int> pos) const{
		return computeIndex(pos.x, pos.y);
	}
	AGPU_CALLABLE_MEMBER T& get(unsigned int x, unsigned int y){
		return data[computeIndex(x, y)];
	}
	AGPU_CALLABLE_MEMBER const T& get(unsigned int x, unsigned int y) const{
		return data[computeIndex(x, y)];
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

template<typename T> class array2DWrapper_view{
public:
	vec2<unsigned int> size;
	const T* data;
	unsigned int length;
	AGPU_CALLABLE_MEMBER array2DWrapper_view() = default;
	AGPU_CALLABLE_MEMBER array2DWrapper_view(const T* d, unsigned int l, vec2<unsigned int> s) : size(s), data(d), length(l){
	}
	AGPU_CALLABLE_MEMBER array2DWrapper_view(const T* d, unsigned int l, unsigned int width, unsigned int height) : array2DWrapper_view(d, l, vec2<unsigned int>{width, height}){}
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(unsigned int x, unsigned int y) const{
		return size.x * y + x;
	}
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(vec2<unsigned int> pos) const{
		return computeIndex(pos.x, pos.y);
	}
	AGPU_CALLABLE_MEMBER const T& get(unsigned int x, unsigned int y) const{
		return data[computeIndex(x, y)];
	}
	AGPU_CALLABLE_MEMBER bool operator==(const array2DWrapper_view& other){
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

template<typename T> class array3DWrapper{
public:
	vec3<unsigned int> size;
	T* data;
	unsigned int length;
	AGPU_CALLABLE_MEMBER array3DWrapper() = default;
	AGPU_CALLABLE_MEMBER array3DWrapper(T* d, unsigned int l, vec3<unsigned int> s) : size(s), data(d), length(l){
	}
	AGPU_CALLABLE_MEMBER array3DWrapper(T* d, unsigned int l, unsigned int width, unsigned int height, unsigned int depth) : array3DWrapper(d, l, vec3<unsigned int>{width, height, depth}){}
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(unsigned int x, unsigned int y, unsigned int z) const{
		return size.x * size.y * z + size.x * y + x;
	}
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(vec3<unsigned int> pos) const{
		return computeIndex(pos.x, pos.y, pos.z);
	}
	AGPU_CALLABLE_MEMBER T& get(unsigned int x, unsigned int y, unsigned int z){
		return data[computeIndex(x, y, z)];
	}
	AGPU_CALLABLE_MEMBER const T& get(unsigned int x, unsigned int y, unsigned int z) const{
		return data[computeIndex(x, y, z)];
	}
	AGPU_CALLABLE_MEMBER bool operator==(const array3DWrapper& other){
		return data == other.data;
	}
	AGPU_CALLABLE_MEMBER T& operator[](vec3<unsigned int> s){
		return get(s.x, s.y, s.z);
	}
	AGPU_CALLABLE_MEMBER const T& operator[](vec3<unsigned int> s) const{
		return get(s.x, s.y, s.z);
	}
	template<class func> AGPU_CALLABLE_MEMBER void foreach(const func& f){
		for(unsigned int i=0;i<size.x;i++){
			for(unsigned int j=0;j<size.y;j++){
				for(unsigned int k=0;k<size.z;k++){
					f(get(i, j, k), i, j, k);
				}
			}
		}
	}
};

template<typename T> class array3DWrapper_view{
public:
	vec3<unsigned int> size;
	const T* data;
	unsigned int length;
	AGPU_CALLABLE_MEMBER array3DWrapper_view() = default;
	AGPU_CALLABLE_MEMBER array3DWrapper_view(const T* d, unsigned int l, vec3<unsigned int> s) : size(s), data(d), length(l){
	}
	AGPU_CALLABLE_MEMBER array3DWrapper_view(const T* d, unsigned int l, unsigned int width, unsigned int height, unsigned int depth) : array3DWrapper_view(d, l, vec3<unsigned int>{width, height, depth}){}
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(unsigned int x, unsigned int y, unsigned int z) const{
		return size.x * size.y * z + size.x * y + x;
	}
	constexpr AGPU_CALLABLE_MEMBER unsigned int computeIndex(vec3<unsigned int> pos) const{
		return computeIndex(pos.x, pos.y, pos.z);
	}
	AGPU_CALLABLE_MEMBER const T& get(unsigned int x, unsigned int y, unsigned int z) const{
		return data[computeIndex(x, y, z)];
	}
	AGPU_CALLABLE_MEMBER bool operator==(const array3DWrapper_view& other){
		return data == other.data;
	}
	AGPU_CALLABLE_MEMBER const T& operator[](vec3<unsigned int> s) const{
		return get(s.x, s.y, s.z);
	}
	template<class func> AGPU_CALLABLE_MEMBER void foreach(const func& f){
		for(unsigned int i=0;i<size.x;i++){
			for(unsigned int j=0;j<size.y;j++){
				for(unsigned int k=0;k<size.z;k++){
					f(get(i, j, k), i, j, k);
				}
			}
		}
	}
};

template<typename T, unsigned int L> struct arrayNDWrapper_t{
	using type = void;
};
template<typename T> struct arrayNDWrapper_t<T, 2>{
	using type = array2DWrapper<T>;
};
template<typename T> struct arrayNDWrapper_t<T, 3>{
	using type = array3DWrapper<T>;
};
template<typename T, unsigned int L> using arrayNDWrapper = arrayNDWrapper_t<T, L>::type;

