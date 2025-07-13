#include "rawDataLoader.hpp"
#include <fftw3.h>
#include <thread>

template<> std::vector<std::vector<fftResult<double>>> rawDataLoader<double>::computePixelTimeFFTs(){
	unsigned int threadCount = std::thread::hardware_concurrency();
	std::vector<double*> in_datas(threadCount);
	for(auto& in_data : in_datas)
		in_data = fftw_alloc_real(data.size());
	std::size_t outputSize = data.size()/2 + 1;
	std::vector<fftwComplex<double>*> out_datas(threadCount);
	for(auto& out_data : out_datas)
		out_data = fftw_alloc_complex(outputSize);

	std::vector<std::vector<unsigned int>> rowsPerThread(threadCount);
	for(unsigned int i=0;i<threadCount;i++){
		for(unsigned int x=i;x<size.x();x+=threadCount){
			rowsPerThread[i].push_back(x);
		}
	}

	std::vector<std::vector<fftResult<double>>> output;
	output.resize(size.x());
	for(auto& vec : output)
		vec.resize(size.y());

	fftw_plan plan = fftw_plan_dft_r2c_1d(data.size(), in_datas[0], out_datas[0], FFTW_PATIENT);
	auto runFFTThread = [&](unsigned int threadNum){
	
		for(const auto& x : rowsPerThread[threadNum]){
			for(unsigned int y=0;y<size.y();y++){
				for(std::size_t i=0;i<data.size();i++){
					in_datas[threadNum][i] = data[i].multi[{x, y}];
				}
				fftw_execute_dft_r2c(plan, in_datas[threadNum], out_datas[threadNum]);
				output[x][y].frequency.resize(data.size()/2 +1);
				output[x][y].magnitude.resize(data.size()/2 +1);
				for(std::size_t i=0;i<=data.size()/2;i++){
					float_t freq = i * sampleRate / data.size();
					float_t mag = std::sqrt(out_datas[threadNum][i][0] * out_datas[threadNum][i][0] + out_datas[threadNum][i][1] * out_datas[threadNum][i][1]);
					output[x][y].frequency[i] = freq;
					output[x][y].magnitude[i] = mag;
				}
			}
			std::cout<<"row "<<x<<" out of "<<size.x()<<" done with ffts"<<std::endl;
		}
	};

	std::vector<std::thread> threads;
	for(unsigned int i=0;i<threadCount;i++)
		threads.push_back(std::thread(runFFTThread, i));
	for(auto& thread : threads)
		thread.join();

	return output;
}

template class rawDataLoader<float>;
template class rawDataLoader<double>;

