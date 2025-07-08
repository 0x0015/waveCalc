#pragma once
#include <sycl/sycl.hpp>

struct stateMembers2D{
	double current;
	double right, left, up, down;
};

//unify this so that the actual logic in the cpu and gpu code should be the same
inline constexpr ACPP_UNIVERSAL_TARGET double calculateUAtPos2D(const stateMembers2D& currentStateVals, double previousStateVal, double partitionSize, double c, double dt, double mu){
	double term1 = currentStateVals.current * (4.0-8.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term2 = previousStateVal * (mu*dt-2.0) / (mu*dt+2.0);
	double term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term3_pt2 = currentStateVals.right + currentStateVals.left + currentStateVals.up + currentStateVals.down;
	double term3 = term3_pt1 * term3_pt2;
	return term1 + term2 + term3;
}


struct stateMembers3D{
	double current;
	double right, left, up, down, above, below;
};

//unify this so that the actual logic in the cpu and gpu code should be the same
inline constexpr ACPP_UNIVERSAL_TARGET double calculateUAtPos3D(const stateMembers3D& currentStateVals, double previousStateVal, double partitionSize, double c, double dt, double mu){
	double term1 = currentStateVals.current * (4.0-12.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term2 = previousStateVal * (mu*dt-2.0) / (mu*dt+2.0);
	double term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term3_pt2 = currentStateVals.right + currentStateVals.left + currentStateVals.up + currentStateVals.down + currentStateVals.above + currentStateVals.below;
	double term3 = term3_pt1 * term3_pt2;
	return term1 + term2 + term3;
}

