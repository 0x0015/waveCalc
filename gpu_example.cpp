#include <iostream>
#include "gpuLib/deviceVector.hpp"
#include "gpuLib/algorithm.hpp"

using namespace hipUtil;

void doSomething(){
	agpuDeviceProp_t prop;
	check_error(agpuGetDeviceProperties(&prop, 0));
	std::cout<<"Using agpu device: "<<prop.name<<std::endl;

	device_vector<unsigned int> a{8, 1, 2 ,5 ,3, 9, 7, 6, 4, 0};
	device_vector<unsigned int> out(a.size());

	sort(a);

	auto a_sorted = a.copy_to_host();
	std::cout<<"Sorted :\t[";
	for(unsigned int i=0;i<a_sorted.size();i++){
		std::cout<<a_sorted[i];
		if(i+1 < a_sorted.size())
			std::cout<<", ";
	}
	std::cout<<"]"<<std::endl;

	fill(a, 42);

	std::cout<<"Filled :\t[";
	for(unsigned int i=0;i<a.size();i++){
		std::cout<<a[i];
		if(i+1 < a.size())
			std::cout<<", ";
	}
	std::cout<<"]"<<std::endl;

	/*
	struct gen_func_t{
		__device__ unsigned int operator()(){
			return 37;
		}
	};
	
	generate(a, gen_func_t{});

	std::cout<<"Generated :\t[";
	for(unsigned int i=0;i<a.size();i++){
		std::cout<<a[i];
		if(i+1 < a.size())
			std::cout<<", ";
	}
	std::cout<<"]"<<std::endl;
	*/

	generate(a, [] __device__ (){return 7;});

	std::cout<<"Lambda generated :\t[";
	for(unsigned int i=0;i<a.size();i++){
		std::cout<<a[i];
		if(i+1 < a.size())
			std::cout<<", ";
	}
	std::cout<<"]"<<std::endl;

	/*
	struct trans_func_t{
		__device__ unsigned int operator()(unsigned int v){
			return v*v;
		}
	};
	transform(a, trans_func_t{});

	std::cout<<"Transformed :\t[";
	for(unsigned int i=0;i<a.size();i++){
		std::cout<<a[i];
		if(i+1 < a.size())
			std::cout<<", ";
	}
	std::cout<<"]"<<std::endl;
	*/

	transform(a, [] __device__ (unsigned int val){return val+1;});

	std::cout<<"Lambda transformed :\t[";
	for(unsigned int i=0;i<a.size();i++){
		std::cout<<a[i];
		if(i+1 < a.size())
			std::cout<<", ";
	}
	std::cout<<"]"<<std::endl;

	unsigned int accum = accumulate(a);

	std::cout<<"Accumulated: "<<accum<<std::endl;

	//not yet supported
	/*
	struct my_vec2{
		float x,y;
	};
	device_vector<my_vec2> b{{4, 1}, {7,3}, {2,2}, {1,4}, {6,7}};

	struct my_vec2_decomposer_t{
		__device__ std::tuple<float&, float&> operator()(my_vec2& val) const{
			return {val.x, val.y};
		}
	};
	sort(b, my_vec2_decomposer_t{});

	std::cout<<"Custom decomposer sort :\t[";
	for(unsigned int i=0;i<b.size();i++){
		std::cout<<"{"<<b[i].x<<", "<<b[i].y<<"}";
		if(i+1 < b.size())
			std::cout<<", ";
	}
	std::cout<<"]"<<std::endl;
	*/
	
	
}
