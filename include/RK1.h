#ifndef __RK1_H__
#define __RK1_H__

#include <iostream>


#include <cuda.h>
#include "complex.h"
#include "timer.h"
#include <thrust/device_vector.h>


#include "constants.h"

template<class Real>
class RK1{
	private:
		tuneParam tp;
		complex *a0;
		complex *a1;
		complex *tmp;
		bool tuning;
		size_t nx;
		size_t ny;
		//bool usetex;
		uint typ;
		std::string name;
	public:
		RK1(){};
		~RK1(){};
		void setup(complex *a0, complex *a1, size_t nx, size_t ny);
		void tune();
		//void run(bool usetex_);

		void run(uint typ_);

		void callKernel(uint3 grid, uint3 block, size_t smem);
		float flops(float time_ms) const;
		long long flop() const;
		long long bytes() const;
		float bwdth(float time_ms) const;
};






#endif
