#pragma once
#include <vector>
#include <fftw3.h>

template<typename float_t> struct fftwComplex_t{
	using type = void;
};
template<> struct fftwComplex_t<float>{
	using type = fftwf_complex;
};
template<> struct fftwComplex_t<double>{
	using type = fftw_complex;
};
template<typename float_t> using fftwComplex = fftwComplex_t<float_t>::type;

template<typename float_t> struct fftResult{
	std::vector<float_t> magnitude;
	std::vector<float_t> frequency;
};

