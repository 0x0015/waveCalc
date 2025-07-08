#pragma once
#include "multiVec.hpp"
#include "imageWriter.hpp"
#include "rawDataWriter.hpp"

template<unsigned int dim> struct chamberDef{
	sycl::vec<double, dim> pos;
	sycl::vec<double, dim> size;
	double c; //wave speed
	double mu; //damping constant
	sycl::vec<unsigned int, dim> pos_internal;
	sycl::vec<unsigned int, dim> size_internal;
	ACPP_UNIVERSAL_TARGET inline bool isPointInChamber(sycl::vec<unsigned int, dim> point) const;
};

template<> ACPP_UNIVERSAL_TARGET inline bool chamberDef<2>::isPointInChamber(sycl::vec<unsigned int, 2> point) const{
	return(point.x() >= pos_internal.x() && point.y() >= pos_internal.y() && point.x() < pos_internal.x() + size_internal.x() && point.y() < pos_internal.y() + size_internal.y());
}
template<> ACPP_UNIVERSAL_TARGET inline bool chamberDef<3>::isPointInChamber(sycl::vec<unsigned int, 3> point) const{
	return(point.x() >= pos_internal.x() && point.y() >= pos_internal.y() && point.z() >= pos_internal.z() && point.x() < pos_internal.x() + size_internal.x() && point.y() < pos_internal.y() + size_internal.y() && point.z() < pos_internal.z() + size_internal.z());
}

enum simulationEdgeMode{
	REFLECT,
	VOID /*wave will just go off edge of sim and dissapear*/
};

template<unsigned int dim> struct waveChamber_t{
	using type = void;
};
template<unsigned int dim> using waveChamber = waveChamber_t<dim>::type;

