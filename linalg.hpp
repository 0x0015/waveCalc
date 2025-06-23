#pragma once
#include "gpuLib/agpuCallableMember.hpp"
#include <cmath>

template<typename T> struct vec2{
	T x, y;
	constexpr AGPU_CALLABLE_MEMBER bool operator==(const vec2& other) const{
		return x==other.x && y==other.y;
	}
	constexpr AGPU_CALLABLE_MEMBER vec2 operator+(const vec2& other) const{
		return vec2{x+other.x, y+other.y};
	}
	constexpr AGPU_CALLABLE_MEMBER vec2 operator-(const vec2& other) const{
		return vec2{x-other.x, y-other.y};
	}
	constexpr AGPU_CALLABLE_MEMBER vec2 operator*(const T& scalar) const{
		return vec2{x * scalar, y * scalar};
	}
	constexpr AGPU_CALLABLE_MEMBER vec2 operator/(const T& scalar) const{
		return vec2{x / scalar, y / scalar};
	}
	constexpr AGPU_CALLABLE_MEMBER T dot(const vec2& other) const{
		return(x*other.x + y*other.y);
	}
	constexpr AGPU_CALLABLE_MEMBER T norm() const{
		return std::sqrt(x*x + y*y);
	}
	template<typename T2> vec2<T2> convert() const{
		return vec2<T2>{(T2)x, (T2)y};
	}
};

template<typename T> constexpr AGPU_CALLABLE_MEMBER vec2<T> operator*(const double& o1, const vec2<T>& o2){
	return o2 * o1;
}

template<typename T> struct vec3{
	T x, y, z;
	constexpr AGPU_CALLABLE_MEMBER bool operator==(const vec3& other) const{
		return x==other.x && y==other.y && z==other.z;
	}
	constexpr AGPU_CALLABLE_MEMBER vec3 operator+(const vec3& other) const{
		return vec3{x+other.x, y+other.y, z+other.z};
	}
	constexpr AGPU_CALLABLE_MEMBER vec3 operator-(const vec3& other) const{
		return vec3{x-other.x, y-other.y, z-other.z};
	}
	constexpr AGPU_CALLABLE_MEMBER vec3 operator*(const T& scalar) const{
		return vec3{x * scalar, y * scalar, z * scalar};
	}
	constexpr AGPU_CALLABLE_MEMBER vec3 operator/(const T& scalar) const{
		return vec3{x / scalar, y / scalar, z / scalar};
	}
	constexpr AGPU_CALLABLE_MEMBER T dot(const vec3& other) const{
		return(x*other.x + y*other.y + z*other.z);
	}
	constexpr AGPU_CALLABLE_MEMBER vec3 cross(const vec3& other) const{
		return vec3{
			y * other.z - z * other.y,
			z * other.x - x * other.z,
			x * other.y - y * other.x
		};
	}
	constexpr AGPU_CALLABLE_MEMBER T norm() const{
		return std::sqrt(x*x + y*y + z*z);
	}
	template<typename T2> vec3<T2> convert() const{
		return vec3<T2>{(T2)x, (T2)y, (T2)z};
	}
};

template<typename T> constexpr AGPU_CALLABLE_MEMBER vec3<T> operator*(const double& o1, const vec3<T>& o2){
	return o2 * o1;
}

