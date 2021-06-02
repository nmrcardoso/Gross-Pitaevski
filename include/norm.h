#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <iostream>


#include <cuda.h>
#include "complex.h"
#include "timer.h"
#include "constants.h"
#include <thrust/device_vector.h>




template<class Real>
struct NormArg{
	complex* a0_d;
	Real* d_sum;
	complex* d_min;
	complex* d_max;
	Real integral;
};




template<class Real>
class Normalize{
	private:
		tuneParam tp;
		bool tuning;
		int ncol;
		Real dx;
		Real dy;
		NormArg<Real> arg;
		size_t nx;
		size_t ny;
		uint typ;
		std::string name;
	void callKernel_gm(uint3 grid, uint3 block, size_t smem);
	void callKernel_tex(uint3 grid, uint3 block, size_t smem);
	public:
		Normalize(){};
		~Normalize(){};
		void setup(complex* a0_d, Real* d_sum, complex* d_min, complex* d_max, int nx_, int ny_, Real dx_, Real dy_);
		void tune();
		void run(Real& integral, complex& minvar, complex& maxvar, uint typ_);
		void callKernel(uint3 grid, uint3 block, size_t smem);
		float flops(float time_ms) const;
		long long flop() const;
		long long bytes() const;
		float bwdth(float time_ms) const;
};


#endif
