#include "waveChamber.hpp"

int main(){
	waveChamber chamber;
	chamber.init({1, 1}, 0.00001, 1, 0.5, 1000, waveChamber::EXECUTION_MODE_GPU);
	chamber.setSinglePoint({500, 500}, 1.0);
	//chamber.setSinglePoint({750, 750}, -0.75);
	//chamber.printuVals();
	chamber.runSimulation(1, 0.01, 0.01);
	//chamber.printuVals();

	return 0;
}
