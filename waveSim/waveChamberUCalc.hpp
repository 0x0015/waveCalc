#pragma once
#include <sycl/sycl.hpp>

template<typename float_t> struct stateMembers2D{
	float_t current;
	float_t right, left, up, down;
};

//unify this so that the actual logic in the cpu and gpu code should be the same
template<typename float_t> inline constexpr float_t calculateUAtPos2D(const stateMembers2D<float_t>& currentStateVals, float_t previousStateVal, float_t partitionSize, float_t c, float_t dt, float_t mu){
	float_t term1 = currentStateVals.current * (4.0-8.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	float_t term2 = previousStateVal * (mu*dt-2.0) / (mu*dt+2.0);
	float_t term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	float_t term3_pt2 = currentStateVals.right + currentStateVals.left + currentStateVals.up + currentStateVals.down;
	float_t term3 = term3_pt1 * term3_pt2;
	return term1 + term2 + term3;
}


template<typename float_t> struct stateMembers3D{
	float_t current;
	float_t right, left, up, down, above, below;
};

//unify this so that the actual logic in the cpu and gpu code should be the same
template<typename float_t> inline constexpr float_t calculateUAtPos3D(const stateMembers3D<float_t>& currentStateVals, float_t previousStateVal, float_t partitionSize, float_t c, float_t dt, float_t mu){
	float_t term1 = currentStateVals.current * (4.0-12.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	float_t term2 = previousStateVal * (mu*dt-2.0) / (mu*dt+2.0);
	float_t term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	float_t term3_pt2 = currentStateVals.right + currentStateVals.left + currentStateVals.up + currentStateVals.down + currentStateVals.above + currentStateVals.below;
	float_t term3 = term3_pt1 * term3_pt2;
	return term1 + term2 + term3;
}

