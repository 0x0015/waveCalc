#include "waveChamber.hpp"
#include <array>

int main(){
	waveChamber<3> chamber;
	//std::array chambers{chamberDef<2>{{0, 0}, {1, 1}, 1, 0.9}};//, chamberDef{{0, 0.9}, {1, 0.1}, 1, 99}};
	std::array chambers{chamberDef<3>{{0, 0, 0}, {1, 1, 1}, 1, 0.9}};//, chamberDef{{0, 0.9}, {1, 0.1}, 1, 99}};
	chamber.init({1.0, 1.0, 1.0}, 0.00001, 250, chambers, VOID);
	chamber.setPointVarryingTimeFunction({0.5, 0.5, 0.5}, [](double t){return std::max(1-t-t, 0.0);});
	//chamber.setSinglePoint({0.75, 0.75}, -0.75);
	chamber.setPointVarryingTimeFunction({0.75, 0.75, 0.75}, [](double t){return std::sin(t * 100);});
	//chamber.printuVals();
	chamber.runSimulation(1, 0.01, 0.01, 0.01);
	//chamber.printuVals();

	return 0;
}
