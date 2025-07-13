#include "waveSim/waveChamber.hpp"
#include "processOutput/rawDataLoader.hpp"
#include <array>
#include <cmath>

int main(){
	rawDataLoader<double> loader;
	loader.loadFile("rawOutput.grid2dD");
	loader.writeOutImages("outputImages", std::nullopt);
	auto ffts = loader.computePixelTimeFFTs();
	for(unsigned int i=0;i<ffts[500][500].frequency.size();i++)
		std::cout<<ffts[500][500].frequency[i]<<" Hz: "<<ffts[500][500].magnitude[i]<<"\n";

	return 0;
}
