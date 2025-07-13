#pragma once
#include <sycl/sycl.hpp>

template<typename T> class array2DWrapper{
public:
	sycl::vec<unsigned int, 2> size;
	T* data;
	unsigned int length;
	array2DWrapper() = default;
	array2DWrapper(T* d, unsigned int l, sycl::vec<unsigned int, 2> s) : size(s), data(d), length(l){
	}
	array2DWrapper(T* d, unsigned int l, unsigned int width, unsigned int height) : array2DWrapper(d, l, sycl::vec<unsigned int, 2>{width, height}){}
	constexpr unsigned int computeIndex(unsigned int x, unsigned int y) const{
		return size.x() * y + x;
	}
	unsigned int computeIndex(sycl::vec<unsigned int, 2> pos) const{
		return computeIndex(pos.x(), pos.y());
	}
	T& get(unsigned int x, unsigned int y){
		return data[computeIndex(x, y)];
	}
	const T& get(unsigned int x, unsigned int y) const{
		return data[computeIndex(x, y)];
	}
	bool operator==(const array2DWrapper& other){
		return data == other.data;
	}
	T& operator[](sycl::vec<unsigned int, 2> s){
		return get(s.x(), s.y());
	}
	const T& operator[](sycl::vec<unsigned int, 2> s) const{
		return get(s.x(), s.y());
	}
	template<class func> void foreach(const func& f){
		for(unsigned int i=0;i<size.x();i++){
			for(unsigned int j=0;j<size.y();j++){
				f(get(i, j), i, j);
			}
		}
	}
};

template<typename T> class array2DWrapper_view{
public:
	sycl::vec<unsigned int, 2> size;
	const T* data;
	unsigned int length;
	array2DWrapper_view() = default;
	array2DWrapper_view(const T* d, unsigned int l, sycl::vec<unsigned int, 2> s) : size(s), data(d), length(l){
	}
	array2DWrapper_view(const T* d, unsigned int l, unsigned int width, unsigned int height) : array2DWrapper_view(d, l, sycl::vec<unsigned int, 2>{width, height}){}
	constexpr unsigned int computeIndex(unsigned int x, unsigned int y) const{
		return size.x() * y + x;
	}
	unsigned int computeIndex(sycl::vec<unsigned int, 2> pos) const{
		return computeIndex(pos.x(), pos.y());
	}
	const T& get(unsigned int x, unsigned int y) const{
		return data[computeIndex(x, y)];
	}
	bool operator==(const array2DWrapper_view& other){
		return data == other.data;
	}
	const T& operator[](sycl::vec<unsigned int, 2> s) const{
		return get(s.x(), s.y());
	}
	template<class func> void foreach(const func& f){
		for(unsigned int i=0;i<size.x();i++){
			for(unsigned int j=0;j<size.y();j++){
				f(get(i, j), i, j);
			}
		}
	}
};

template<typename T> class array3DWrapper{
public:
	sycl::vec<unsigned int, 3> size;
	T* data;
	unsigned int length;
	array3DWrapper() = default;
	array3DWrapper(T* d, unsigned int l, sycl::vec<unsigned int, 3> s) : size(s), data(d), length(l){
	}
	array3DWrapper(T* d, unsigned int l, unsigned int width, unsigned int height, unsigned int depth) : array3DWrapper(d, l, sycl::vec<unsigned int, 3>{width, height, depth}){}
	constexpr unsigned int computeIndex(unsigned int x, unsigned int y, unsigned int z) const{
		return size.x() * size.y() * z + size.x() * y + x;
	}
	unsigned int computeIndex(sycl::vec<unsigned int, 3> pos) const{
		return computeIndex(pos.x(), pos.y(), pos.z());
	}
	T& get(unsigned int x, unsigned int y, unsigned int z){
		return data[computeIndex(x, y, z)];
	}
	const T& get(unsigned int x, unsigned int y, unsigned int z) const{
		return data[computeIndex(x, y, z)];
	}
	bool operator==(const array3DWrapper& other){
		return data == other.data;
	}
	T& operator[](sycl::vec<unsigned int, 3> s){
		return get(s.x(), s.y(), s.z());
	}
	const T& operator[](sycl::vec<unsigned int, 3> s) const{
		return get(s.x(), s.y(), s.z());
	}
	template<class func> void foreach(const func& f){
		for(unsigned int i=0;i<size.x();i++){
			for(unsigned int j=0;j<size.y();j++){
				for(unsigned int k=0;k<size.z();k++){
					f(get(i, j, k), i, j, k);
				}
			}
		}
	}
};

template<typename T> class array3DWrapper_view{
public:
	sycl::vec<unsigned int, 3> size;
	const T* data;
	unsigned int length;
	array3DWrapper_view() = default;
	array3DWrapper_view(const T* d, unsigned int l, sycl::vec<unsigned int, 3> s) : size(s), data(d), length(l){
	}
	array3DWrapper_view(const T* d, unsigned int l, unsigned int width, unsigned int height, unsigned int depth) : array3DWrapper_view(d, l, sycl::vec<unsigned int, 3>{width, height, depth}){}
	constexpr unsigned int computeIndex(unsigned int x, unsigned int y, unsigned int z) const{
		return size.x() * size.y() * z + size.x() * y + x;
	}
	unsigned int computeIndex(sycl::vec<unsigned int, 3> pos) const{
		return computeIndex(pos.x(), pos.y(), pos.z());
	}
	const T& get(unsigned int x, unsigned int y, unsigned int z) const{
		return data[computeIndex(x, y, z)];
	}
	bool operator==(const array3DWrapper_view& other){
		return data == other.data;
	}
	const T& operator[](sycl::vec<unsigned int, 3> s) const{
		return get(s.x(), s.y(), s.z());
	}
	template<class func> void foreach(const func& f){
		for(unsigned int i=0;i<size.x();i++){
			for(unsigned int j=0;j<size.y();j++){
				for(unsigned int k=0;k<size.z();k++){
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

