


#include <iostream>

#include <SDL2/SDL.h>

#include <string>
#include <stdexcept>
#include <cstdio>


#include <cuda.h>
#include "cuda_common.h"
#include <cuda_runtime_api.h>


//https://nvlabs.github.io/cub/
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include "complex.h"
#include "cuda_common.h"
#include "constants.h"
#include "draw.h"


#include <typeinfo>

#define BLOCK_I 64
#define BLOCK_J 3
#define BLOCK_II 32
#define BLOCK_JJ 8
#define I2D(ni,i,j) (((ni)*(j)) + i)

#define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })



#include "textures.h"
#include "functions.h"
#include "device.h"



// CUDA kernel to fill plot_rgba_data array for plotting
template<class Real>
__global__ void get_rgba_kernel (int pitch, int ncol, complex minvar, complex scale,
                                 complex *plot_data,
                                 unsigned int *plot_rgba_data,
                                 unsigned int *cmap_rgba_data, int posx, int posy, int barx, int bardim, int pts, bool plotlegend){
    uint i, j, i2d, icol;
    Real frac;    
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
	complex in = complex::zero();
	uint i3d = posx + i + (posy + ny-1 -j) * pitch;
	if(j<ny && i<nx)  in = plot_data[i + j*nx];
	if(j<ny && i<nx){	
		frac = (in.abs2()-minvar.real()) * scale.real();
		icol = (int)(frac * (Real)ncol);
		plot_rgba_data[i3d] = cmap_rgba_data[icol];

		frac = (in.arg()-minvar.imag()) * scale.imag();
		icol = (int)(frac * (Real)ncol);
		plot_rgba_data[pitch/2 + i3d] = cmap_rgba_data[icol];


		if(plotlegend){
			
			i2d = posx + nx + barx + i + (j + posy)*pitch;

			if(i < bardim) {
				frac = (Real)(ny-1 - j) / (Real)(ny-1);
				icol = (int)(frac * (Real)ncol);
				plot_rgba_data[i2d] = cmap_rgba_data[icol]; 
				plot_rgba_data[pitch / 2 + i2d] = cmap_rgba_data[icol]; 
			}
			int length = bardim/8;
			if( (i < length) || (i >= (bardim - length) && i < bardim) ) {
				Real step = (Real)(ny-1)/(Real)(pts-1);
				for(uint ii=0;ii<pts;ii++){
					uint res = ii * step;
					if( j == res ){
						plot_rgba_data[i2d] = 0;
						plot_rgba_data[pitch / 2 + i2d] = 0;
					}
				}
			}

			if( i < length ){
				Real step = (Real)(ny-1)/(Real)(pts-1);
				for(uint ii=1;ii<pts-1;ii++){
					uint res = ii * step;
					if( j == res ){
						plot_rgba_data[i3d] = 0;
						plot_rgba_data[pitch / 2 + i3d] = 0;
						plot_rgba_data[i3d + nx - length] = 0;
						plot_rgba_data[pitch / 2 + nx - length + i3d] = 0;
					}
					
				}	
			}
			if( j < length ){
				Real step = (Real)(nx-1)/(Real)(pts-1);
				for(uint ii=1;ii<pts-1;ii++){
					uint res = ii * step;
					if( i == res ){
						plot_rgba_data[i3d] = 0;
						plot_rgba_data[pitch / 2 + i3d] = 0;
					}
					
				}	
			}

			if( j >= ny - length ){
				Real step = (Real)(nx-1)/(Real)(pts-1);
				for(uint ii=1;ii<pts-1;ii++){
					uint res = ii * step;
					if( i == res ){
						plot_rgba_data[i3d] = 0;
						plot_rgba_data[pitch / 2 + i3d] = 0;
					}
					
				}	
			}
		}

	}
}





template <class Real>
void MapColor<Real>::setup(unsigned int *cmap_rgba_data_, int ncol_, size_t nx_, size_t ny_, int posx_, int posy_, int barx_, int bardim_, int pts_, bool plotlegend_) {
	cmap_rgba_data = cmap_rgba_data_;
	posx = posx_;
	posy = posy_;
	barx = barx_;
	pts = pts_;
	plotlegend = plotlegend_;
	bardim = bardim_;
	ncol=ncol_;
	nx=nx_;
	ny=ny_;
	tuning = true;
	tp.block = make_uint3(32,1,1);
	tp.grid = make_uint3((nx+tp.block.x-1)/tp.block.x,(ny+tp.block.y-1)/tp.block.y, 1);
}
template <class Real>
void MapColor<Real>::tune(){			
	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	

	uint3 block = make_uint3(32,1,1);
	uint3 grid = make_uint3((nx+block.x-1)/block.x,(ny+block.y-1)/block.y, 1);
	size_t shared_bytes = 0;
	tp.time = 9999999999.0;
	name = typeid(*this).name();
	cudaError_t error;
    cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float elapsed_time;
	while(tuning){
		cudaDeviceSynchronize();
		cudaGetLastError(); // clear error counter
        cudaEventRecord(start, 0);
		callKernel(grid, block, shared_bytes); 
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, start, end);
        cudaDeviceSynchronize();
        error = cudaGetLastError();


		{ // check that error state is cleared
		cudaDeviceSynchronize();
		cudaError_t error1 = cudaGetLastError();
		if (error1 != cudaSuccess){
			printf("Failed to clear error state %s\n", cudaGetErrorString(error1));
			exit(1);
		}
		}
		if( tp.time > elapsed_time  && (error == cudaSuccess) ){
			tp.block = block;
			tp.grid = grid;
			tp.shared_bytes = 0;
			tp.time = elapsed_time;
		}
		if(getVerbosity() == DEBUG_VERBOSE){
			if( (error == cudaSuccess)  ) std::cout << name << ": block=(" << block.x << "," << block.y << ")=" << block.x*block.y << ", grid=(" << grid.x << "," << grid.y << "), smem=" << shared_bytes/(1024.) << " KB, time=" << elapsed_time << " ms, " << flops(elapsed_time) << " Gflop/s, " << bwdth(elapsed_time) << " GB/s" << std::endl;
			else std::cout << name << ": block=(" << block.x << "," << block.y << ")=" << block.x*block.y << ", grid=(" << grid.x << "," << grid.y << "), smem=" << shared_bytes/(1024.) << " KB, error: " << cudaGetErrorString(error) << std::endl;
		}
		block.x += 32;
		int blocksize = block.x * block.y;
		if( block.x == deviceProp.maxThreadsDim[0] || blocksize > deviceProp.maxThreadsPerBlock){
			block.x = 32;
			block.y += 1;
			blocksize = block.x * block.y;
		}
		grid = make_uint3((nx+block.x-1)/block.x,(ny+block.y-1)/block.y, 1);
		if(block.x > deviceProp.maxThreadsDim[0] || grid.x > deviceProp.maxGridSize[0] || blocksize > deviceProp.maxThreadsPerBlock) tuning = false;
		if(block.y > deviceProp.maxThreadsDim[1] || grid.y > deviceProp.maxGridSize[1] || blocksize > deviceProp.maxThreadsPerBlock) tuning = false;
	}
	CUDA_SAFE_CALL(cudaEventDestroy( start));
	CUDA_SAFE_CALL(cudaEventDestroy( end));
	tuning = false;
	if( getVerbosity() >= SUMMARIZE ) std::cout << name << ": block=(" << tp.block.x << "," << tp.block.y << ")=" << tp.block.x*tp.block.y << ", grid=(" << tp.grid.x << "," << tp.grid.y << "), smem=" << tp.shared_bytes/(1024.) << " KB, time=" << tp.time << " ms, " << flops(tp.time) << " Gflop/s, " << bwdth(tp.time) << " GB/s\n" << std::endl;
}
template <class Real>
void MapColor<Real>::run(complex *data_, unsigned int *plot_rgba_data_, complex minvar_, complex scale_, size_t pitch_){
	data=data_;
	plot_rgba_data=plot_rgba_data_;
	minvar=minvar_;
	scale=scale_;
	pitch=pitch_;
	if(tuning) tune();
	callKernel(tp.grid, tp.block, tp.shared_bytes);
    CUDA_SAFE_THREAD_SYNC();
    CUT_CHECK_ERROR("get_rgba failed.");
}
template <class Real>
void MapColor<Real>::callKernel(uint3 grid, uint3 block, size_t smem){
    get_rgba_kernel<Real><<<grid, block, smem, 0>>>(pitch, ncol, minvar, scale, data, plot_rgba_data, cmap_rgba_data, posx, posy, barx, bardim, pts, plotlegend);
}
	
		
template <class Real>
float MapColor<Real>::flops(float time_ms) const{
	return flop() * 1.0e-6 / time_ms;	
}

template <class Real>
long long MapColor<Real>::flop() const{
	return 6. * (nx) * (ny);	
}
template <class Real>
long long MapColor<Real>::bytes() const{
	return (4 * sizeof(uint) + sizeof(complex)) * nx*ny;	
}
template <class Real>
float MapColor<Real>::bwdth(float time_ms) const{
	return bytes() * 1.0e-6 / time_ms;	
}

template class MapColor<float>;
template class MapColor<double>;


