#pragma once
#include <vector>
#include <glm/glm.hpp>

template<typename T> class vector2D{
public:
	glm::uvec2 size;
	std::vector<T> data;
	vector2D() = default;
	vector2D(glm::uvec2 s) : size(s){
		data.resize(s.x * s.y);
	}
	vector2D(unsigned int width, unsigned int height) : vector2D(glm::uvec2{width, height}){}
	T& get(unsigned int x, unsigned int y){
		return data[size.x * y + x];
	}
	const T& get(unsigned int x, unsigned int y) const{
		return data[size.x * y + x];
	}
	bool operator==(const vector2D& other){
		return data == other.data;
	}
	T& operator[](glm::uvec2 s){
		return get(s.x, s.y);
	}
	const T& operator[](glm::uvec2 s) const{
		return get(s.x, s.y);
	}
	template<class func> void foreach(const func& f){
		for(unsigned int i=0;i<size.x;i++){
			for(unsigned int j=0;j<size.y;j++){
				f(get(i, j), i, j);
			}
		}
	}
};
