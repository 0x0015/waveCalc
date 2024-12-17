#pragma once
#include "linalg.hpp"
#include "gpuLib/agpuCallableMember.hpp"
#include "generalContainer.hpp"
#include <memory>
#include "globalExecution.hpp"

struct materialProps{
	double c;
	double mu;
};

struct waveRectGeometry{
	vec2<unsigned int> pos; //in partitions
	vec2<unsigned int> size; //in partitions
	materialProps material;
	unsigned int calculatedStartOffset;
};

class waveEnvironment{
public:
	double dx; //partition size
	unsigned int activeStateNum;
	std::shared_ptr<generalContainer<waveRectGeometry>> geometry;
	std::shared_ptr<generalContainer<double>> state;
	void init(const std::vector<waveRectGeometry>& geometry);
	void calculateStartOffsets();
private:
	void init_cpu(const std::vector<waveRectGeometry>& geometry);
	void calculateStartOffsets_cpu();
	void init_gpu(const std::vector<waveRectGeometry>& geometry);
	void calculateStartOffsets_gpu();
};

class waveEnvironmentAccessor{
public:
	waveRectGeometry* geometry;
	unsigned int geometryLength;
	double* state;
	unsigned int stateLength; //3 will be stored so multiply by 3
	unsigned int activeStateNum;
	inline AGPU_CALLABLE_MEMBER std::pair<vec2<unsigned int>, vec2<unsigned int>> getBounds(){
		if(geometryLength == 0){
			return {};
		}
		vec2<unsigned int> pos = geometry[0].pos;
		vec2<unsigned int> size = geometry[0].size;
		for(unsigned int i=1;i<geometryLength;i++){
			pos.x = std::min(pos.x, geometry[i].pos.x);
			pos.y = std::min(pos.y, geometry[i].pos.y);
			size.x = std::max(size.x, geometry[i].size.x);
			size.y = std::max(size.y, geometry[i].size.y);
		}
		return {pos, size};
	}
	inline AGPU_CALLABLE_MEMBER constexpr unsigned int getStateNumInNStates(unsigned int n){
		return (activeStateNum + n) % 3;
	}
	static inline AGPU_CALLABLE_MEMBER constexpr unsigned int getIndexFromCoordsInRect(vec2<unsigned int> coords, vec2<unsigned int> rectSize){
		return coords.y * rectSize.x + coords.x;
	}
	static inline AGPU_CALLABLE_MEMBER constexpr vec2<unsigned int> getCoordsFromIndexInRect(unsigned int index, vec2<unsigned int> rectSize){
		return {index % rectSize.x, index / rectSize.x};
	}
	static inline AGPU_CALLABLE_MEMBER constexpr bool isPointInRect(vec2<unsigned int> point, vec2<unsigned int> rectPos, vec2<unsigned int> rectSize){
		return point.x >= rectPos.x && point.y >= rectPos.y && point.x < rectPos.x + rectSize.x && point.y < rectPos.y + rectSize.y;
	}
	inline AGPU_CALLABLE_MEMBER int getGeometryIndexContainingPoint(vec2<unsigned int> point) const{
		for(unsigned int i=0;i<geometryLength;i++){
			if(isPointInRect(point, geometry[i].pos, geometry[i].size))
				return i + stateLength * activeStateNum;
		}
		return -1;
	}
	inline AGPU_CALLABLE_MEMBER unsigned int getGeometryIndexFromStateIndex(unsigned int index) const{
		for(unsigned int i=0;i<geometryLength;i++){
			if(geometry[i].calculatedStartOffset > index)
				return i-1;
		}
		return geometryLength-1;
	}
	inline AGPU_CALLABLE_MEMBER vec2<unsigned int> getPosFromStateIndex(unsigned int index) const{
		unsigned int geometryIndex = getGeometryIndexFromStateIndex(index);
		return getCoordsFromIndexInRect(index - geometry[geometryIndex].calculatedStartOffset, geometry[geometryIndex].size);
	}
	AGPU_CALLABLE_MEMBER const double get(unsigned int x, unsigned int y, unsigned int partitionsInPast = 0) const{
		int index = getGeometryIndexContainingPoint({x, y}) + stateLength * ((activeStateNum + 3 - partitionsInPast) % 3);
		if(index < 0){
			return 0;
		}
		return state[geometry[index].calculatedStartOffset + getIndexFromCoordsInRect({x, y}, geometry[index].size)];
	}
	AGPU_CALLABLE_MEMBER const double operator[](vec2<unsigned int> s) const{
		return get(s.x, s.y);
	}
	AGPU_CALLABLE_MEMBER const void set(unsigned int x, unsigned int y, double val){
		int index = getGeometryIndexContainingPoint({x, y});
		if(index < 0){
			return;
		}
		state[geometry[index].calculatedStartOffset + getIndexFromCoordsInRect({x, y}, geometry[index].size)] = val;
	}
};

