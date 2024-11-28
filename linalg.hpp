#pragma once
#include "gpuLib/agpuCallableMember.hpp"

template<typename T> struct vec2{
	T x, y;
	AGPU_CALLABLE_MEMBER bool operator==(const vec2& other) const{
		return x==other.x && y==other.y;
	}
};

template<typename T> struct vec3{
	T x, y, z;
	AGPU_CALLABLE_MEMBER bool operator==(const vec3& other) const{
		return x==other.x && y==other.y && z==other.z;
	}
};
