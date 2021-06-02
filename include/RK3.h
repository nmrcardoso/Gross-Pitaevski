#ifndef __RK3_H__
#define __RK3_H__

#include <iostream>


#include <cuda.h>
#include "complex.h"
#include "timer.h"
#include <thrust/device_vector.h>

#include "constants.h"



template<class Real>
struct RK3Arg{
	complex* a0;
	complex* a1;
	complex* a2;
	Real* d_sum;
};

/*
template<class Real>
class RK3{
	private:
		tuneParam tp;
		complex* tmp;
		RK3Arg<Real> arg;
		bool tuning;
		size_t nx;
		size_t ny;
		bool usetex;
	public:
		RK3(){};
		~RK3(){};
		void setup(complex *a0, complex *a1, complex *a2, Real* d_sum, size_t nx, size_t ny);
		void tune();
		void run(bool usetex_);
		void callKernel();
		float flops(float time_ms) const;
		long long flop() const;
		long long bytes() const;
		float bwdth(float time_ms) const;
};*/





template<class Real>
class RK3{
	private:
		tuneParam tp;
		complex *a0;
		complex *a1;
		complex *a2;
		complex *tmp;
		RK3Arg<Real> arg;
		bool tuning;
		size_t nx;
		size_t ny;
		//bool usetex;
		uint typ;
		std::string name;
	public:
		RK3(){};
		~RK3(){};
		void setup(complex *a0_, complex *a1_, complex *a2_, Real* d_sum_, size_t nx_, size_t ny_);
		void tune();
		void run(uint typ_);
		void callKernel(uint3 grid, uint3 block, size_t smem);
		float flops(float time_ms) const;
		long long flop() const;
		long long bytes() const;
		float bwdth(float time_ms) const;
};




#endif
