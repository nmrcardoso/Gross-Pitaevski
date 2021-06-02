
#include <iostream>

#include <SDL2/SDL.h>

#include <string>
#include <stdexcept>
#include <cstdio>


#include <cuda.h>
#include "cuda_common.h"
#include <cuda_runtime_api.h>



#include "complex.h"
#include "cuda_common.h"
#include "RK2.h"
#include "constants.h"

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



/*
template<class T, class T2>
__global__ void rungekutta_step2_smem(complex *a0, complex *a1, complex *a2){

  complex *smem0 = SharedMemory<T,T2>();

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int id = i + j * n;
  int it = threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x+2);
  
  if(i<n && j<n){
    smem0[it] = getdata1(a1, id);
    //__syncthreads();
  }

  if(i>0 && j>0 && i<n-1 && j<n-1){
    if(threadIdx.x == 0) smem0[it-1] = getdata1(a1, id-1);
    if(threadIdx.y == 0) smem0[index(0,-1)] = getdata1(a1, id-n);
    if(threadIdx.x == blockDim.x - 1) smem0[it+1] = getdata1(a1, id+1);
    if(threadIdx.y == blockDim.y - 1) smem0[index(0,1)] = getdata1(a1, id+n);
    __syncthreads();
    T px = (T)0.5 / dx2<T>();
    T py = (T)0.5 / dx<T>();
    complex temp = smem0[it+1] - smem0[it] * (T)4.0 + smem0[it-1];
    complex temp0 = (smem0[it+1] - smem0[it-1]) * py;

    temp = temp + smem0[index(0,1)] + smem0[index(0,-1)];
    complex temp1  = (smem0[index(0,1)] - smem0[index(0,-1)]) * py;
    temp = temp * px;

    px = (T)i * dx<T>() - (T)10.0;
    py = (T)j * dx<T>() - (T)10.0;

    temp0 = temp1 * px  - temp0 * py;
    px = ( (px * px + py * py) * (T)0.5) + (smem0[it].abs2() * g<T>());
    temp1 =  smem0[it] * px;

    temp1 = temp1 - temp - Iomega<T,T2>() * temp0;

    temp = getdata0(a0, id);
    temp1 = temp1 * InvIminusGamma<T,T2>();           
    a2[id] = temp * ((T)0.75) + smem0[it] * (T)0.25 + temp1 * (dt<T>() * (T)0.25);

  }

}
*/


template<class Real, bool usetex>
__global__ void rungekutta_step2(complex *a0, complex *a1, complex *a2){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i>0 && j>0 && i<nx-1 && j<ny-1){
		int id = i + j * nx;

		complex firstdx = ELEM_LOAD<usetex, 1, Real>(a1, id + 1);
		complex res = ELEM_LOAD<usetex, 1, Real>(a1, id - 1);
		complex actual = ELEM_LOAD<usetex, 1, Real>(a1, id);
		complex seconddxy = (firstdx - actual * 2.0 + res) * 0.5 * oneoverdx2<Real>();
		firstdx = (firstdx - res) * 0.5 * oneoverdx<Real>();

		complex firstdy = ELEM_LOAD<usetex, 1, Real>(a1, id + nx);
		res = ELEM_LOAD<usetex, 1, Real>(a1, id - nx);
		seconddxy += (firstdy - actual * 2.0 + res) * 0.5 * oneoverdy2<Real>();
		firstdy  = (firstdy - res) * 0.5 * oneoverdy<Real>();


		Real px = i * dx<Real>() - Lx<Real>();
		Real py = j * dy<Real>() - Ly<Real>();

		firstdx = firstdy * px  - firstdx * py;
		px = ( (px * px + py * py) * 0.5) + (actual.abs2() * g<Real>());
		res =  actual * px;

		res = res - seconddxy - Iomega<Real>() * firstdx;
		seconddxy = ELEM_LOAD<usetex, 0, Real>(a0, id);

		res = res * InvIminusGamma<Real>();            
		a2[id] = seconddxy * 0.75 + actual * 0.25 + res * (dt<Real>() * 0.25);
	}
}



template <class Real>
void RK2<Real>::setup(complex *a0_, complex *a1_, complex *a2_, size_t nx_, size_t ny_) {
	//tune();
	a0 = a0_;
	a1=a1_;
	a2=a2_;
	nx = nx_;
	ny=ny_;
	tuning = true;
	tp.block = make_uint3(32,1,1);
	tp.grid = make_uint3((nx+tp.block.x-1)/tp.block.x,(ny+tp.block.y-1)/tp.block.y, 1);
	usetex = false;
}
template <class Real>
void RK2<Real>::tune(){			
	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	
	uint3 block = make_uint3(32,1,1);
	uint3 grid = make_uint3((nx+block.x-1)/block.x,(ny+block.y-1)/block.y, 1);
	tp.time = 9999999999.0;
	size_t size = nx*ny * sizeof(complex);
	CUDA_SAFE_CALL(cudaMalloc(&tmp, size));
	CUDA_SAFE_CALL(cudaMemcpy(tmp, a2, size, cudaMemcpyDeviceToDevice));
	cudaError_t error;
    cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float elapsed_time;
	while(tuning){
		cudaDeviceSynchronize();
		cudaGetLastError(); // clear error counter
        cudaEventRecord(start, 0);
		callKernel(); 
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
		if( (error == cudaSuccess) ) std::cout << typeid(*this).name() << ": block=(" << block.x << "," << block.y << ")=" << block.x*block.y << ", grid=(" << grid.x << "," << grid.y << "), time=" << elapsed_time << " ms, " << flops(elapsed_time) << " Gflop/s, " << bwdth(elapsed_time) << " GB/s" << std::endl;
		else std::cout << typeid(*this).name() << ": block=(" << block.x << "," << block.y << ")=" << block.x*block.y << ", grid=(" << grid.x << "," << grid.y << "), error: " << cudaGetErrorString(error) << std::endl;
		
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
	CUDA_SAFE_CALL(cudaMemcpy(a2, tmp, size, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaFree(tmp));
	std::cout << typeid(*this).name() << ": block=(" << tp.block.x << "," << tp.block.y << ")=" << tp.block.x*tp.block.y << ", grid=(" << tp.grid.x << "," << tp.grid.y << "), time=" << tp.time << " ms, " << flops(tp.time) << " Gflop/s, " << bwdth(tp.time) << " GB/s\n" << std::endl;
}
template <class Real>
void RK2<Real>::run(bool usetex_){
	usetex = usetex_;
	if(tuning) tune();
	callKernel();
    CUDA_SAFE_THREAD_SYNC();
    CUT_CHECK_ERROR("RK2 failed.");
}
template <class Real>
void RK2<Real>::callKernel(){
	if(usetex) rungekutta_step2<Real, true><<<tp.grid,tp.block>>>(a0, a1, a2); 
	else rungekutta_step2<Real, false><<<tp.grid,tp.block>>>(a0, a1, a2); 
}
	
		
template <class Real>
float RK2<Real>::flops(float time_ms) const{
	return flop() * 1.0e-6 / time_ms;	
}
template <class Real>
long long RK2<Real>::flop() const{
	return 81 * (nx-2) * (ny-2);	
}
template <class Real>
long long RK2<Real>::bytes() const{
	return 7 * (nx-2) * (ny-2) * sizeof(complex);
}
template <class Real>
float RK2<Real>::bwdth(float time_ms) const{
	return bytes() * 1.0e-6 / time_ms;	
}
	
template class RK2<float>;
template class RK2<double>;



