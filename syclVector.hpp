#pragma once
#include <sycl/sycl.hpp>
#include <vector>
#include <span>

template<class T> class deviceVector{
public:
	using value_type = T;
	deviceVector(sycl::queue& queue) : q(queue){
		ptr = nullptr;
		internal_size = 0;
		internal_capacity = 0;
	}
	deviceVector(sycl::queue& queue, size_t default_size) : q(queue){
		ptr = sycl::malloc_device<T>(default_size, q);
		internal_capacity = default_size;
		internal_size = default_size;
	}
	deviceVector(sycl::queue& queue, size_t default_size, const T& v) : q(queue){
		ptr = sycl::malloc_device<T>(default_size, q);
		q.fill(ptr, v, default_size).wait();

		internal_capacity = default_size;
		internal_size = default_size;
	}
	deviceVector(sycl::queue& queue, const std::initializer_list<T>& i_list) : q(queue){
		int sz = i_list.size();

		ptr = sycl::malloc_device<T>(sz, q);
		q.memcpy(ptr, std::data(i_list), sizeof(T) * sz).wait();

		internal_size = sz;
		internal_capacity = sz;
	}
	deviceVector(sycl::queue& queue, std::span<const T>& vec) : q(queue){
		int sz = vec.size();

		ptr = sycl::malloc_device<T>(sz, q);
		q.memcpy(ptr, vec.data(), sizeof(T) * sz).wait();

		internal_size = sz;
		internal_capacity = sz;
	}
	deviceVector(sycl::queue& queue, const std::vector<T>& vec) : q(queue){
		int sz = vec.size();

		ptr = sycl::malloc_device<T>(sz, q);
		q.memcpy(ptr, vec.data(), sizeof(T) * sz).wait();

		internal_size = sz;
		internal_capacity = sz;
	}
	template<size_t S> deviceVector(sycl::queue& queue, const std::array<T, S>& vec) : q(queue){
		constexpr size_t sz = S;

		ptr = sycl::malloc_device<T>(sz, q);
		q.memcpy(ptr, vec.data(), sizeof(T) * sz).wait();

		internal_size = sz;
		internal_capacity = sz;
	}
	deviceVector(sycl::queue& queue, const deviceVector<T>& vec) : q(queue){
		int sz = vec.size();

		ptr = sycl::malloc_device<T>(sz, q);
		q.memcpy(ptr, vec.data(), sizeof(T) * sz).wait();

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
		q.memcpy(output.data(), ptr, internal_size * sizeof(T)).wait();
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
	deviceVector& operator=(const deviceVector& other){
		if(this == &other)
			return *this;

		if(other.size() > internal_capacity){
			internal_capacity = other.size();
			reallocate();
		}

		q.memcpy(ptr, other.data(), other.size() * sizeof(T));
		q.wait();
		internal_size = other.size();

		return *this;
	}
	deviceVector& operator=(const std::vector<T>& other){

		if(other.size() > internal_capacity){
			internal_capacity = other.size();
			reallocate();
		}

		q.memcpy(ptr, other.data(), other.size() * sizeof(T)).wait();
		internal_size = other.size();

		return *this;
	}
	operator T*(){
		return ptr;
	}
	operator const T*() const{
		return ptr;
	}
	~deviceVector(){
		if(ptr == nullptr)
			return;
		sycl::free(ptr, q);
		q.wait();
	}
private:
	T* ptr;
	size_t internal_size;
	size_t internal_capacity;
	sycl::queue& q;
	void reallocate(){
		T* temp = sycl::malloc_device<T>(internal_capacity, q);
		q.memcpy(temp, ptr, internal_size * sizeof(T)).wait();
		sycl::free(ptr, q);
		ptr = temp;
	}
};

