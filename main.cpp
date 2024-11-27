#include "waveChamber.hpp"

void doSomething();

int main(){
	doSomething();
	return 0;
	waveChamber chamber;
	chamber.init({1, 1}, 0.001, 1, 0.5, 1000);
	chamber.currentState->uVals[{500, 500}] = 1.0;
	chamber.printuVals();
	for(unsigned int i=0;i<200;i++){
		chamber.step();
		//chamber.printuVals();
		chamber.writeToImage("outputImages/image" + std::to_string(i) + ".png", 1.0);
	}

	return 0;
}
