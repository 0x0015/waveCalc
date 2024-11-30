#pragma once
#include "gpuLib/agpuCallableMember.hpp"
#include "linalg.hpp"

struct stateMembers{
	double current;
	double right, left, above, below;
};

//unify this so that the actual logic in the cpu and gpu code should be the same
inline constexpr AGPU_CALLABLE_MEMBER double calculateUAtPos(vec2<unsigned int> pos, stateMembers currentStateVals, double previousStateVal, double partitionSize, double c, double dt, double mu){
	double term1 = currentStateVals.current * (4.0-8.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term2 = previousStateVal * (mu*dt-2.0) / (mu*dt+2.0);
	double term3_pt1 = (2.0*c*c*dt*dt/(partitionSize*partitionSize))/(mu*dt+2.0);
	double term3_pt2 = currentStateVals.right + currentStateVals.left + currentStateVals.above + currentStateVals.below;
	double term3 = term3_pt1 * term3_pt2;
	return term1 + term2 + term3;
}

