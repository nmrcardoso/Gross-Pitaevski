
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
#include "RK1.h"

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


__device__ inline int indexx(int i, int j){

  return threadIdx.x + 1 + i + (threadIdx.y + 1 + j) * (blockDim.x+2);
}

template<class Real, bool usetex>
__global__ void rungekutta_step1_smem(complex *a0, complex *a1){

  complex *smem = SharedMemory<complex>();

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int id = i + j * nx;
  int it = threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x+2);
  
  if(i<nx && j<ny) smem[it] = ELEM_LOAD<usetex, 0, Real>(a0, id);
 

  if(i>0 && j>0 && i<nx-1 && j<ny-1){
    if(threadIdx.x == 0) smem[it-1] = ELEM_LOAD<usetex, 0, Real>(a0, id-1);
    if(threadIdx.y == 0) smem[index(0,-1)] = ELEM_LOAD<usetex, 0, Real>(a0, id-nx);
    if(threadIdx.x == blockDim.x - 1) smem[it+1] = ELEM_LOAD<usetex, 0, Real>(a0, id+1);
    if(threadIdx.y == blockDim.y - 1) smem[index(0,1)] = ELEM_LOAD<usetex, 0, Real>(a0, id+nx);  
	}   
    __syncthreads();


  if(i>0 && j>0 && i<nx-1 && j<ny-1){

	complex seconddxy = (smem[it+1] - smem[it] * 2.0 + smem[it-1]) * 0.5 * oneoverdx2<Real>();
	complex firstdx = (smem[it+1] - smem[it-1]) * 0.5 * oneoverdx<Real>();

	seconddxy += (smem[indexx(0,1)] - smem[it] * 2.0 + smem[indexx(0,-1)]) * 0.5 * oneoverdy2<Real>();
	complex firstdy  = (smem[indexx(0,1)] - smem[indexx(0,-1)]) * 0.5 * oneoverdy<Real>();


	Real px = i * dx<Real>() - Lx<Real>();
	Real py = j * dy<Real>() - Ly<Real>();

	firstdx = firstdy * px  - firstdx * py;
	px = ( (px * px + py * py) * 0.5) + (smem[it].abs2() * g<Real>());
	complex res =  smem[it] * px;

	res = res - seconddxy - Iomega<Real>() * firstdx;;

	res = res * InvIminusGamma<Real>();            
	a1[id] = smem[it] + res * dt<Real>();
	}
}


template<class Real, bool usetex>
__global__ void rungekutta_step1(complex *a0, complex *a1){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i>0 && j>0 && i<nx-1 && j<ny-1){
		int id = i + j * nx;

		complex firstdx = ELEM_LOAD<usetex, 0, Real>(a0, id + 1);
		complex res = ELEM_LOAD<usetex, 0, Real>(a0, id - 1);

		complex actual = ELEM_LOAD<usetex, 0, Real>(a0, id);
		complex seconddxy = (firstdx - actual * 2.0 + res) * 0.5 * oneoverdx2<Real>();
		firstdx = (firstdx - res) * 0.5 * oneoverdx<Real>();

		complex firstdy = ELEM_LOAD<usetex, 0, Real>(a0, id + nx);
		res = ELEM_LOAD<usetex, 0, Real>(a0, id - nx);
		seconddxy += (firstdy - actual * 2.0 + res) * 0.5 * oneoverdy2<Real>();
		firstdy  = (firstdy - res) * 0.5 * oneoverdy<Real>();


		Real px = i * dx<Real>() - Lx<Real>();
		Real py = j * dy<Real>() - Ly<Real>();

		firstdx = firstdy * px  - firstdx * py;
		px = ( (px * px + py * py) * 0.5) + (actual.abs2() * g<Real>());
		res =  actual * px;

		res = res - seconddxy - Iomega<Real>() * firstdx;;

		res = res * InvIminusGamma<Real>();            
		a1[id] = actual + res * dt<Real>();
  }
}





template <class Real>
void RK1<Real>::setup(complex *a0_, complex *a1_, size_t nx_, size_t ny_) {
	//tune();
	a0 = a0_;
	a1=a1_;
	nx = nx_;
	ny=ny_;
	tuning = true;
	tp.block = make_uint3(32,1,1);
	tp.grid = make_uint3((nx+tp.block.x-1)/tp.block.x,(ny+tp.block.y-1)/tp.block.y, 1);
	//usetex = false;
	typ = 0;
}
template <class Real>
void RK1<Real>::tune(){			
	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	
	//Timer t0;
	uint3 block = make_uint3(32,1,1);
	uint3 grid = make_uint3((nx+block.x-1)/block.x,(ny+block.y-1)/block.y, 1);
	size_t shared_bytes = 0;
	if(typ>=2) shared_bytes = (block.x + 2)*(block.y+2) * sizeof(complex);
	tp.time = 9999999999.0;
	size_t size = nx*ny * sizeof(complex);
	CUDA_SAFE_CALL(cudaMalloc(&tmp, size));
	CUDA_SAFE_CALL(cudaMemcpy(tmp, a1, size, cudaMemcpyDeviceToDevice));
	cudaError_t error;
    cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float elapsed_time;
	name = typeid(*this).name();
	name += "_" + ToString(typ);
	while(tuning){
		cudaDeviceSynchronize();
		cudaGetLastError(); // clear error counter
		//t0.start();
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
			tp.shared_bytes = shared_bytes;
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
		if(typ>=2) shared_bytes = (block.x + 2)*(block.y+2) * sizeof(complex);
		grid = make_uint3((nx+block.x-1)/block.x,(ny+block.y-1)/block.y, 1);
		if(block.x > deviceProp.maxThreadsDim[0] || grid.x > deviceProp.maxGridSize[0] || blocksize > deviceProp.maxThreadsPerBlock) tuning = false;
		if(block.y > deviceProp.maxThreadsDim[1] || grid.y > deviceProp.maxGridSize[1] || blocksize > deviceProp.maxThreadsPerBlock) tuning = false;
	}
	CUDA_SAFE_CALL(cudaEventDestroy( start));
	CUDA_SAFE_CALL(cudaEventDestroy( end));
	tuning = false;
	CUDA_SAFE_CALL(cudaMemcpy(a1, tmp, size, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaFree(tmp));
	if( getVerbosity() >= SUMMARIZE ) std::cout << name << ": block=(" << tp.block.x << "," << tp.block.y << ")=" << tp.block.x*tp.block.y << ", grid=(" << tp.grid.x << "," << tp.grid.y << "), smem=" << tp.shared_bytes/(1024.) << " KB, time=" << tp.time << " ms, " << flops(tp.time) << " Gflop/s, " << bwdth(tp.time) << " GB/s\n" << std::endl;
}
template <class Real>
//void RK1<Real>::run(bool usetex_){
void RK1<Real>::run(uint typ_){
	//usetex = usetex_;
	typ = typ_;
	std::string name1 = typeid(*this).name();
	name1 += "_" + ToString(typ);
	if(tuning || name != name1){
		tuning = true;
		tune();
	}
	callKernel(tp.grid, tp.block, tp.shared_bytes);
    CUDA_SAFE_THREAD_SYNC();
    CUT_CHECK_ERROR("RK1 failed.");
}
template <class Real>
void RK1<Real>::callKernel(uint3 grid, uint3 block, size_t smem){
	switch(typ){
		case 0:
			rungekutta_step1<Real, false><<<grid, block, smem, 0>>>(a0, a1);
		break;
		case 1:
			rungekutta_step1<Real, true><<<grid, block, smem, 0>>>(a0, a1);
		break;
		case 2:
			rungekutta_step1_smem<Real, false><<<grid, block, smem, 0>>>(a0, a1);
		break;
		case 3:
			rungekutta_step1_smem<Real, true><<<grid, block, smem, 0>>>(a0, a1);
		break;
		default:
			rungekutta_step1<Real, false><<<grid, block, smem, 0>>>(a0, a1);
		break;
	}
	//if(usetex) rungekutta_step1<Real, true><<<tp.grid,tp.block>>>(a0, a1);  
	//else rungekutta_step1<Real, false><<<tp.grid,tp.block>>>(a0, a1); 
}
		
template <class Real>
float RK1<Real>::flops(float time_ms) const{
	return flop() * 1.0e-6 / time_ms;	
}
template <class Real>
long long RK1<Real>::flop() const{
	return 74 * (nx-2) * (ny-2);	
}
template <class Real>
long long RK1<Real>::bytes() const{
	return 6 * (nx-2) * (ny-2) * sizeof(complex);	
}
template <class Real>
float RK1<Real>::bwdth(float time_ms) const{
	return bytes() * 1.0e-6 / time_ms;	
}
	
template class RK1<float>;
template class RK1<double>;







