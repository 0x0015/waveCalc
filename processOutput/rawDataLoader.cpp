#include "rawDataLoader.hpp"
#include <fstream>
#include <filesystem>

template<typename float_t> void rawDataLoader<float_t>::loadFile(std::string_view filename){
	if(!std::filesystem::exists(filename)){
		std::cout<<"Error: unable to load raw data file \""<<filename<<"\" as the file does not exist"<<std::endl;
		return;
	}
	auto fileSize = std::filesystem::file_size(filename);
	std::ifstream file((std::string)filename, std::ios_base::in | std::ios_base::binary);
	std::vector<uint8_t> fileContents(fileSize);
	file.read((char*)fileContents.data(), fileContents.size());
	file.close();

	constexpr unsigned int extraPadding = 32;
	constexpr std::string_view checkString_true = "2dDGridFile";
	for(unsigned int i=0;i<checkString_true.size();i++){
		if(std::bit_cast<char>(fileContents[i]) != checkString_true[i]){
			std::cout<<"Error: raw file did not match 2dDGridFile header type"<<std::endl;
			return;
		}
	}
	std::size_t currentReadOffset = checkString_true.size();
	fileVersionNum = *(unsigned int*)(fileContents.data() + currentReadOffset);
	currentReadOffset += sizeof(unsigned int);
	size.x() = *(unsigned int*)(fileContents.data() + currentReadOffset);
	currentReadOffset += sizeof(unsigned int);
	size.y() = *(unsigned int*)(fileContents.data() + currentReadOffset);
	currentReadOffset += sizeof(unsigned int);
	std::cout<<"rawDataLoader got the grid size to be "<<size.x()<<" x "<<size.y()<<std::endl;
	sampleRate = *(float_t*)(fileContents.data() + currentReadOffset);
	currentReadOffset += sizeof(float_t);
	currentReadOffset += extraPadding;

	std::size_t bytesLeft = fileSize - currentReadOffset;
	std::size_t dataChunkSize = size.x() * size.y() * sizeof(float_t);
	std::size_t chunksLeft = bytesLeft / dataChunkSize;
	std::cout<<"Found "<<chunksLeft<<" chunks of data in file.  Loading..."<<std::endl;

	for(unsigned int i=0;i<chunksLeft;i++){
		data.push_back({});
		loadedData& back = data.back();
		back.raw_data.resize(size.x() * size.y());
		std::memcpy(back.raw_data.data(), fileContents.data() + currentReadOffset, dataChunkSize);
		back.multi = array2DWrapper<float_t>(back.raw_data.data(), back.raw_data.size(), size);
		currentReadOffset += dataChunkSize;
	}
}

template class rawDataLoader<float>;
template class rawDataLoader<double>;

