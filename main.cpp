#include "waveChamber.hpp"
#include <array>

int main(){
	waveChamber<EXECUTION_MODE_CPU> chamber;
	std::array<decltype(chamber)::chamberDef, 1> chambers{decltype(chamber)::chamberDef{{0, 0}, {1, 1}, 1, 0.5}};
	chamber.init({1, 1}, 0.00001, 1000, chambers);
	chamber.setSinglePoint({500, 500}, 1.0);
	chamber.setSinglePoint({750, 750}, -0.75);
	//chamber.printuVals();
	chamber.runSimulation(1, 0.01, 0.01, 0.01);
	//chamber.printuVals();

	return 0;
}
