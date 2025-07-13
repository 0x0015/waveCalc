#pragma once
#include <string_view>
#include <optional>
#include <sycl/sycl.hpp>
#include "fftProcess.hpp"
#include "../waveSim/multiVec.hpp"

template<typename float_t> class rawDataLoader{
public:
	sycl::vec<unsigned int, 2> size;
	struct loadedData{
		std::vector<float_t> raw_data;
		array2DWrapper<float_t> multi;
	};
	std::vector<loadedData> data;
	uint32_t fileVersionNum;
	float_t sampleRate;
	void loadFile(std::string_view filename);
	void writeOutImages(std::string_view imageFolder, std::optional<float_t> maximum /*leave blank to auto detect.  Note costly*/);
	std::vector<std::vector<fftResult<float_t>>> computePixelTimeFFTs();
};

