#ifndef __DRAW_H__
#define __DRAW_H__

#include <iostream>


#include <cuda.h>
#include "complex.h"
#include "timer.h"
#include "constants.h"
#include <thrust/device_vector.h>






template<class Real>
class MapColor{
	private:
		tuneParam tp;
		bool tuning;
		size_t pitch;
		int ncol;
		complex minvar;
		complex scale;
		complex *data;
		unsigned int *plot_rgba_data;
		unsigned int *cmap_rgba_data;
		size_t nx;
		size_t ny;
		std::string name;
		int posx;
		int posy;
		int barx;
		int bardim;
		int pts;
		bool plotlegend;
	public:
		MapColor(){};
		~MapColor(){};
		void setup(unsigned int *cmap_rgba_data_, int ncol_, size_t nx_, size_t ny_, int posx_, int posy_, int barx_, int bardim_, int pts_, bool plotlegend_);
		void tune();
		void run(complex *data_, unsigned int *plot_rgba_data_, complex minvar_, complex maxvar_, size_t pitch_);
		void callKernel(uint3 grid, uint3 block, size_t smem);
		float flops(float time_ms) const;
		long long flop() const;
		long long bytes() const;
		float bwdth(float time_ms) const;
};











#endif
