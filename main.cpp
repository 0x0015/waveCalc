#include "waveChamber.hpp"

int main(){
	waveChamber chamber;
	chamber.init({1, 1}, 0.001, 1, 0.5, 1000, waveChamber::EXECUTION_MODE_GPU);
	chamber.setSinglePoint({500, 500}, 1.0);
	chamber.printuVals();
	chamber.writeToImage("outputImages/image_initial.png", 1.0);
	for(unsigned int i=0;i<100;i++){
		chamber.step();
		chamber.writeToImage("outputImages/image" + std::to_string(i) + ".png", 1.0);
	}
	chamber.printuVals();

	return 0;
}
