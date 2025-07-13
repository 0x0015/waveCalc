#include "rawDataLoader.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "../waveSim/multiVec.hpp"
#include <iostream>

template<typename float_t> void rawDataLoader<float_t>::writeOutImages(std::string_view outputFolder, std::optional<float_t> maximum){
	std::atomic<float_t> expectedMax;
	if(maximum)
		expectedMax = *maximum;
	else{
		//no maximum given, we want to then go through and find it
		expectedMax = 0;
#pragma omp parallel for
		for(unsigned int i=0;i<data.size();i++){
			const auto& dat = data[i];
			for(const auto& val : dat.raw_data){
				float_t absVal = std::abs(val);
				if(absVal > expectedMax)
					expectedMax = absVal;
			}
		}

		std::cout<<"Note: Calculated max cell value for writing out images to be "<<expectedMax<<std::endl;
	}

#pragma omp parallel for
	for(unsigned int i=0;i<data.size();i++){
		const auto& dat = data[i];
		unsigned int x=0;
		unsigned int y=0;
		std::vector<uint8_t> imageData(dat.multi.size.x() * dat.multi.size.y()*3);
		for(unsigned int i=0;i<imageData.size();i+=3){
			float_t val = dat.multi[{x, y}];
			float_t scaled = (val) / expectedMax * 255;
			imageData[i+1] = 0;
			if(scaled < 0){
				imageData[i] = -scaled;
				imageData[i+2] = 0;
			}else{
				imageData[i] = 0;
				imageData[i+2] = scaled;
			}
			x++;
			if(x >= dat.multi.size.x()){
				x = 0;
				y++;
			}
		}

		stbi_write_png((std::string(outputFolder) + "/image" + std::to_string(i) + ".png").c_str(), dat.multi.size.x(), dat.multi.size.y(), 3, imageData.data(), dat.multi.size.x() * 3);
	}
}

template class rawDataLoader<float>;
template class rawDataLoader<double>;

