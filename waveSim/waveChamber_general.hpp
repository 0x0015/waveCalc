#pragma once
#include "multiVec.hpp"
#include "rawDataWriter.hpp"

template<typename T, unsigned int dim> struct chamberDef{
	sycl::vec<T, dim> pos;
	sycl::vec<T, dim> size;
	T c; //wave speed
	T mu; //damping constant
	sycl::vec<unsigned int, dim> pos_internal;
	sycl::vec<unsigned int, dim> size_internal;
	inline bool isPointInChamber(sycl::vec<unsigned int, dim> point) const;
};
template<typename T> struct chamberDef<T, 2>{
	static constexpr unsigned int dim = 2;
	sycl::vec<T, dim> pos;
	sycl::vec<T, dim> size;
	T c; //wave speed
	T mu; //damping constant
	sycl::vec<unsigned int, dim> pos_internal;
	sycl::vec<unsigned int, dim> size_internal;
	inline bool isPointInChamber(sycl::vec<unsigned int, dim> point) const;
};
template<typename T> struct chamberDef<T, 3>{
	static constexpr unsigned int dim = 3;
	sycl::vec<T, dim> pos;
	sycl::vec<T, dim> size;
	T c; //wave speed
	T mu; //damping constant
	sycl::vec<unsigned int, dim> pos_internal;
	sycl::vec<unsigned int, dim> size_internal;
	inline bool isPointInChamber(sycl::vec<unsigned int, dim> point) const;
};

template<typename T> inline bool chamberDef<T, 2>::isPointInChamber(sycl::vec<unsigned int, 2> point) const{
	return(point.x() >= pos_internal.x() && point.y() >= pos_internal.y() && point.x() < pos_internal.x() + size_internal.x() && point.y() < pos_internal.y() + size_internal.y());
}
template<typename T> inline bool chamberDef<T, 3>::isPointInChamber(sycl::vec<unsigned int, 3> point) const{
	return(point.x() >= pos_internal.x() && point.y() >= pos_internal.y() && point.z() >= pos_internal.z() && point.x() < pos_internal.x() + size_internal.x() && point.y() < pos_internal.y() + size_internal.y() && point.z() < pos_internal.z() + size_internal.z());
}

enum simulationEdgeMode{
	REFLECT,
	VOID /*wave will just go off edge of sim and dissapear*/
};

template<typename T, unsigned int dim> struct waveChamber_t{
	using type = void;
};
template<typename T, unsigned int dim> using waveChamber = waveChamber_t<T, dim>::type;

