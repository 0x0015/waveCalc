#pragma once
#include "multiVec.hpp"
#include <array>
#include <string>
#include <memory>
#include <functional>
#include "executionMode.hpp"
#include "generalContainer.hpp"
#include "imageWriter.hpp"
#include "rawDataWriter.hpp"

template<unsigned int dim> struct chamberDef{
	vec<double, dim> pos;
	vec<double, dim> size;
	double c; //wave speed
	double mu; //damping constant
	vec<unsigned int, dim> pos_internal;
	vec<unsigned int, dim> size_internal;
	AGPU_CALLABLE_MEMBER inline constexpr bool isPointInChamber(vec<unsigned int, dim> point) const;
};

template<> AGPU_CALLABLE_MEMBER inline constexpr bool chamberDef<2>::isPointInChamber(vec<unsigned int, 2> point) const{
	return(point.x >= pos_internal.x && point.y >= pos_internal.y && point.x < pos_internal.x + size_internal.x && point.y < pos_internal.y + size_internal.y);
}
template<> AGPU_CALLABLE_MEMBER inline constexpr bool chamberDef<3>::isPointInChamber(vec<unsigned int, 3> point) const{
	return(point.x >= pos_internal.x && point.y >= pos_internal.y && point.z >= pos_internal.z && point.x < pos_internal.x + size_internal.x && point.y < pos_internal.y + size_internal.y && point.z < pos_internal.z + size_internal.z);
}

enum simulationEdgeMode{
	REFLECT,
	VOID /*wave will just go off edge of sim and dissapear*/
};

template<EXECUTION_MODE exMode, unsigned int dim> struct waveChamber_t{
	using type = void;
};
template<EXECUTION_MODE exMode, unsigned int dim> using waveChamber = waveChamber_t<exMode, dim>::type;

