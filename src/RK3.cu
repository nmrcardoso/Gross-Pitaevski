
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
#include "RK3.h"
#include "constants.h"

#include <typeinfo>


//https://nvlabs.github.io/cub/
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>


#include "textures.h"
#include "functions.h"
#include "launch_kernel.cuh"

#include "device.h"










__device__ inline int indexx(int i, int j){

  return threadIdx.x + 1 + i + (threadIdx.y + 1 + j) * (blockDim.x+2);
}






template<class Real, bool usetex>
__global__ void rungekutta_step3_smem(RK3Arg<Real> arg){ 

	complex *smem = SharedMemory<complex>();

	uint id = threadIdx.x + blockIdx.x * blockDim.x;
	uint i = id % nx;
	uint j = id / nx;
	id = i + j * nx;
	int it = threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x+2);

	if(i<nx && j<ny) smem[it] = ELEM_LOAD<usetex, 2, Real>(arg.a2, id);

	if(i>0 && j>0 && i<nx-1 && j<ny-1){
		if(threadIdx.x == 0) smem[it-1] = ELEM_LOAD<usetex, 2, Real>(arg.a2, id-1);
		if(threadIdx.y == 0) smem[index(0,-1)] = ELEM_LOAD<usetex, 2, Real>(arg.a2, id-nx);
		if(threadIdx.x == blockDim.x - 1) smem[it+1] = ELEM_LOAD<usetex, 2, Real>(arg.a2, id+1);
		if(threadIdx.y == blockDim.y - 1) smem[index(0,1)] = ELEM_LOAD<usetex, 2, Real>(arg.a2, id+nx);     
	}    
	__syncthreads();


	Real val = 0.0;
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

		seconddxy = ELEM_LOAD<usetex, 0, Real>(arg.a0, id);

		res = res * InvIminusGamma<Real>();  
		px = 2.0 / 3.0;
		py = 1.0/ 3.0;          
		res = seconddxy * py + smem[it] * px + res * (dt<Real>() * px);
		arg.a0[id] = res;
		val = res.abs2();
	}

  __syncthreads();
  Real *smem0 = (Real*)smem;

  smem0[threadIdx.x] = val;
  __syncthreads();

  //Only one active warp do the reduction
  //Bocks are always multiple of the warp size!
  if(blockDim.x > WARP_SIZE && threadIdx.x < WARP_SIZE){
    for(uint s = 1; s < blockDim.x / WARP_SIZE; s++)
      smem0[threadIdx.x] += smem0[threadIdx.x + WARP_SIZE * s]; 
  }
  //__syncthreads(); //No need to synchronize inside warp!!!!
  //One thread do the warp reduction
  if(threadIdx.x == 0 ) {
    Real sum = smem0[0];
    for(uint s = 1; s < WARP_SIZE; s++) 
      sum += smem0[s];
    CudaAtomicAdd(arg.d_sum, sum);
  }
}




template<class Real, bool usetex>
__global__ void rungekutta_step3(RK3Arg<Real> arg){ 

	uint id = threadIdx.x + blockIdx.x * blockDim.x;
	uint i = id % nx;
	uint j = id / nx;

	Real val = 0.0;
	if(i>0 && j>0 && i<nx-1 && j<ny-1){
		id = i + j * nx;            
		complex firstdx = ELEM_LOAD<usetex, 2, Real>(arg.a2, id + 1);
		complex res = ELEM_LOAD<usetex, 2, Real>(arg.a2, id - 1);
		complex actual = ELEM_LOAD<usetex, 2, Real>(arg.a2, id);
		complex seconddxy = (firstdx - actual * 2.0 + res) * 0.5 * oneoverdx2<Real>();
		firstdx = (firstdx - res) * 0.5 * oneoverdx<Real>();

		complex firstdy = ELEM_LOAD<usetex, 2, Real>(arg.a2, id + nx);
		res = ELEM_LOAD<usetex, 2, Real>(arg.a2, id - nx);
		seconddxy += (firstdy - actual * 2.0 + res) * 0.5 * oneoverdy2<Real>();
		firstdy  = (firstdy - res) * 0.5 * oneoverdy<Real>();


		Real px = i * dx<Real>() - Lx<Real>();
		Real py = j * dy<Real>() - Ly<Real>();

		firstdx = firstdy * px  - firstdx * py;
		px = ( (px * px + py * py) * 0.5) + (actual.abs2() * g<Real>());
		res =  actual * px;

		res = res - seconddxy - Iomega<Real>() * firstdx;
		seconddxy = ELEM_LOAD<usetex, 0, Real>(arg.a0, id);
		res = res * InvIminusGamma<Real>(); 
		px = 2.0 / 3.0;
		py = 1.0/ 3.0;          
		actual = seconddxy * py + actual * px + res * (dt<Real>() * px);
		arg.a0[id] = actual;
		val = actual.abs2();
	}

  Real *smem = SharedMemory<Real>();

  smem[threadIdx.x] = val;
  __syncthreads();

  //Only one active warp do the reduction
  //Bocks are always multiple of the warp size!
  if(blockDim.x > WARP_SIZE && threadIdx.x < WARP_SIZE){
    for(uint s = 1; s < blockDim.x / WARP_SIZE; s++)
      smem[threadIdx.x] += smem[threadIdx.x + WARP_SIZE * s]; 
  }
  //__syncthreads(); //No need to synchronize inside warp!!!!
  //One thread do the warp reduction
  if(threadIdx.x == 0 ) {
    Real sum = smem[0];
    for(uint s = 1; s < WARP_SIZE; s++) 
      sum += smem[s];
    CudaAtomicAdd(arg.d_sum, sum);
  }
}

















template <class Real>
void RK3<Real>::setup(complex *a0_, complex *a1_, complex *a2_, Real* d_sum, size_t nx_, size_t ny_) {
	arg.a0 = a0_;
	arg.a1 = a1_;
	arg.a2 = a2_;
	arg.d_sum = d_sum;
	nx = nx_;
	ny = ny_;
	tuning = true;
	tp.block = make_uint3(32,1,1);
	tp.grid = make_uint3((nx*ny+tp.block.x-1)/tp.block.x, 1, 1);
	typ = 0;
}
template <class Real>
void RK3<Real>::tune(){			
	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	
	//Timer t0;
	uint3 block = make_uint3(32,1,1);
	uint3 grid = make_uint3((nx*ny+block.x-1)/block.x, 1, 1);
	size_t shared_bytes = 0;
	if(typ>=2) shared_bytes = (block.x + 2)*(block.y+2) * sizeof(complex);
	else shared_bytes = (block.x)*(block.y) * sizeof(Real);
	tp.time = 9999999999.0;
	size_t size = nx*ny * sizeof(complex);
	CUDA_SAFE_CALL(cudaMalloc(&tmp, size));
	CUDA_SAFE_CALL(cudaMemcpy(tmp, arg.a0, size, cudaMemcpyDeviceToDevice));
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
		if(typ>=2) shared_bytes = (block.x + 2)*(block.y+2) * sizeof(complex);
		else shared_bytes = (block.x)*(block.y) * sizeof(Real);
		grid = make_uint3((nx*ny+block.x-1)/block.x, 1, 1);
		if(block.x > deviceProp.maxThreadsDim[0] || grid.x > deviceProp.maxGridSize[0] || blocksize > deviceProp.maxThreadsPerBlock) tuning = false;
		if(block.y > deviceProp.maxThreadsDim[1] || grid.y > deviceProp.maxGridSize[1] || blocksize > deviceProp.maxThreadsPerBlock) tuning = false;
	}
	CUDA_SAFE_CALL(cudaEventDestroy( start));
	CUDA_SAFE_CALL(cudaEventDestroy( end));
	tuning = false;
	CUDA_SAFE_CALL(cudaMemcpy(arg.a0, tmp, size, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaFree(tmp));
	if( getVerbosity() >= SUMMARIZE ) std::cout << name << ": block=(" << tp.block.x << "," << tp.block.y << ")=" << tp.block.x*tp.block.y << ", grid=(" << tp.grid.x << "," << tp.grid.y << "), smem=" << tp.shared_bytes/(1024.) << " KB, time=" << tp.time << " ms, " << flops(tp.time) << " Gflop/s, " << bwdth(tp.time) << " GB/s\n" << std::endl;
}
template <class Real>
void RK3<Real>::run(uint typ_){
	typ = typ_;
	std::string name1 = typeid(*this).name();
	name1 += "_" + ToString(typ);
	if(tuning || name != name1){
		tuning = true;
		tune();
	}
    CUDA_SAFE_CALL(cudaMemset(arg.d_sum, 0, sizeof(Real)));
	callKernel(tp.grid, tp.block, tp.shared_bytes);
    CUDA_SAFE_THREAD_SYNC();
    CUT_CHECK_ERROR("RK3 failed.");
}
template <class Real>
void RK3<Real>::callKernel(uint3 grid, uint3 block, size_t smem){
	switch(typ){
		case 0:
			rungekutta_step3<Real, false><<<grid, block, smem, 0>>>(arg);
		break;
		case 1:
			rungekutta_step3<Real, true><<<grid, block, smem, 0>>>(arg);
		break;
		case 2:
			rungekutta_step3_smem<Real, false><<<grid, block, smem, 0>>>(arg);
		break;
		case 3:
			rungekutta_step3_smem<Real, true><<<grid, block, smem, 0>>>(arg);
		break;
		default:
			rungekutta_step3<Real, false><<<grid, block, smem, 0>>>(arg);
		break;
	}
}
		
template <class Real>
float RK3<Real>::flops(float time_ms) const{
	return flop() * 1.0e-6 / time_ms;	
}
template <class Real>
long long RK3<Real>::flop() const{
	return 86LL * (nx-2) * (ny-2);	
}
template <class Real>
long long RK3<Real>::bytes() const{
	return 7LL * (nx-2) * (ny-2) * sizeof(complex);	
}
template <class Real>
float RK3<Real>::bwdth(float time_ms) const{
	return bytes() * 1.0e-6 / time_ms;	
}
	
template class RK3<float>;
template class RK3<double>;








