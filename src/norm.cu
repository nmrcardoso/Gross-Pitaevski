
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
#include "constants.h"
#include "norm.h"

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
#include "launch_kernel.cuh"



#include "device.h"





template< class T, typename ReductionOp>
inline __host__ __device__ T Reduce( T a, T b, ReductionOp reduction_op){
	return reduction_op(a, b);
}

template<class Real, bool Normalize, bool usetex>    
__global__ void Normalize_GetMaxMin(NormArg<Real> arg){

	uint id = threadIdx.x + blockIdx.x * blockDim.x;
	uint i = id % nx;
	uint j = id / nx;

	complex val = complex::zero();
	if(i>0 && j>0 && i<nx-1 && j<ny-1){
	//if(i<n && j<n){
	  id = i + j * nx;
	  complex value = ELEM_LOAD<usetex, 0, Real>(arg.a0_d, id);
	  if(Normalize){
		  value *= arg.integral;
		  arg.a0_d[id]=value;
	  }
	  val = complex::make_complex(value.abs2(),value.arg());
	}

	complex *smem = SharedMemory<complex>();

	Real *smemf = (Real*)smem;
	smemf[threadIdx.x] = val.real();
	__syncthreads();

	//Only one active warp do the reduction
	//Bocks are always multiple of the warp size!
	if(blockDim.x > WARP_SIZE && threadIdx.x < WARP_SIZE){
		for(uint s = 1; s < blockDim.x / WARP_SIZE; s++)
			smemf[threadIdx.x] += smemf[threadIdx.x + WARP_SIZE * s]; 
	}
	//__syncthreads(); //No need to synchronize inside warp!!!!
	//One thread do the warp reduction
	if(threadIdx.x == 0 ) {
		Real sum = smemf[0];
		for(uint s = 1; s < WARP_SIZE; s++) 
			sum += smemf[s];
		CudaAtomicAdd(arg.d_sum, sum);
	}





	__syncthreads();
	smem[threadIdx.x] = val;
	__syncthreads();

	//Only one active warp do the reduction
	//Bocks are always multiple of the warp size!
	if(blockDim.x > WARP_SIZE && threadIdx.x < WARP_SIZE){
		for(uint s = 1; s < blockDim.x / WARP_SIZE; s++)
			smem[threadIdx.x] = Reduce( smem[threadIdx.x], smem[threadIdx.x + WARP_SIZE * s], MinVal<complex>() ); 
	}
	//__syncthreads(); //No need to synchronize inside warp!!!!
	//One thread do the warp reduction
	if(threadIdx.x == 0 ) {
		complex res = smem[0];
		for(uint s = 1; s < WARP_SIZE; s++) 
			res = Reduce( res, smem[s], MinVal<complex>() );
		CudaAtomicMin(arg.d_min, res);
	}



	__syncthreads();
	smem[threadIdx.x] = val;
	__syncthreads();

	//Only one active warp do the reduction
	//Bocks are always multiple of the warp size!
	if(blockDim.x > WARP_SIZE && threadIdx.x < WARP_SIZE){
		for(uint s = 1; s < blockDim.x / WARP_SIZE; s++)
			smem[threadIdx.x] = Reduce( smem[threadIdx.x], smem[threadIdx.x + WARP_SIZE * s], MaxVal<complex>() ); 
	}
	//__syncthreads(); //No need to synchronize inside warp!!!!
	//One thread do the warp reduction
	if(threadIdx.x == 0 ) {
		complex res = smem[0];
		for(uint s = 1; s < WARP_SIZE; s++) 
			res = Reduce( res, smem[s], MaxVal<complex>() );
		CudaAtomicMax(arg.d_max, res);
	}
}







template <class Real>
void Normalize<Real>::setup(complex* a0_d, Real* d_sum, complex* d_min, complex* d_max, int nx_, int ny_, Real dx_, Real dy_) {

	arg.a0_d = a0_d;
	arg.d_sum = d_sum;
	arg.d_min = d_min;
	arg.d_max = d_max;
	dx = dx_;
	dy = dy_;
	nx=nx_;
	ny=ny_;
	tuning = true;
	tp.block = make_uint3(32,1,1);
	tp.grid = make_uint3((nx*ny+tp.block.x-1)/tp.block.x, 1, 1);
	tp.shared_bytes = 0;
	typ = 0;
}


template <class Real>
void Normalize<Real>::tune(){	
	complex *tmp;		
	size_t size = nx*ny * sizeof(complex);
	CUDA_SAFE_CALL(cudaMalloc(&tmp, size));
	CUDA_SAFE_CALL(cudaMemcpy(tmp, arg.a0_d, size, cudaMemcpyDeviceToDevice));

	CUDA_SAFE_CALL(cudaGetDevice( &dev));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	
	uint3 block = make_uint3(32,1,1);
	uint3 grid = make_uint3((nx*ny+block.x-1)/block.x, 1, 1);
	size_t shared_bytes = (block.x)*(block.y) * sizeof(complex);
	tp.time = 9999999999.0;
	name = typeid(*this).name();
	name += "_" + ToString(typ);
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
			tp.shared_bytes = shared_bytes;
			tp.time = elapsed_time;
		}
		if(getVerbosity() == DEBUG_VERBOSE){
			if( (error == cudaSuccess)  ) std::cout << name << ": block=(" << block.x << "," << block.y << ")=" << block.x*block.y << ", grid=(" << grid.x << "," << grid.y << "), smem=" << shared_bytes/(1024.) << " KB, time=" << elapsed_time << " ms, " << flops(elapsed_time) << " Gflop/s, " << bwdth(elapsed_time) << " GB/s" << std::endl;
			else std::cout << name << ": block=(" << block.x << "," << block.y << ")=" << block.x*block.y << ", grid=(" << grid.x << "," << grid.y << "), smem=" << shared_bytes/(1024.) << " KB, error: " << cudaGetErrorString(error) << std::endl;
		}
		
		block.x += 32;
		shared_bytes = (block.x)*(block.y) * sizeof(complex);
		grid = make_uint3((nx*ny+block.x-1)/block.x, 1, 1);
		if(block.x > MAXBLOCKDIMX || grid.x > deviceProp.maxGridSize[0] ) tuning = false;
	}
	CUDA_SAFE_CALL(cudaEventDestroy( start));
	CUDA_SAFE_CALL(cudaEventDestroy( end));
	tuning = false;
	CUDA_SAFE_CALL(cudaMemcpy(arg.a0_d, tmp, size, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaFree(tmp));  
	if( getVerbosity() >= SUMMARIZE ) std::cout << name << ": block=(" << tp.block.x << "," << tp.block.y << ")=" << tp.block.x*tp.block.y << ", grid=(" << tp.grid.x << "," << tp.grid.y << "), smem=" << tp.shared_bytes/(1024.) << " KB, time=" << tp.time << " ms, " << flops(tp.time) << " Gflop/s, " << bwdth(tp.time) << " GB/s\n" << std::endl;
}

template <class Real>
void Normalize<Real>::run(Real& integral, complex& minvar, complex& maxvar, uint typ_){
	typ = typ_;
	std::string name1 = typeid(*this).name();
	name1 += "_" + ToString(typ);
	if(tuning || name != name1){
		tuning = true;
		tune();
	}
	CUDA_SAFE_CALL(cudaMemcpy(&arg.integral, arg.d_sum, sizeof(Real), cudaMemcpyDeviceToHost));
	arg.integral = 1.0 / sqrt(arg.integral * dx * dy);
	//std::cout << "integral0=" << arg.integral << std::endl;

	if(tuning) tune();
    CUDA_SAFE_CALL(cudaMemset(arg.d_sum, 0, sizeof(Real))); 
    CUDA_SAFE_CALL(cudaMemset(arg.d_min, 0, sizeof(complex))); 
    CUDA_SAFE_CALL(cudaMemset(arg.d_max, 0, sizeof(complex))); 
	callKernel(tp.grid, tp.block, tp.shared_bytes);
    CUDA_SAFE_THREAD_SYNC();
    CUT_CHECK_ERROR("Norm failed.");
	CUDA_SAFE_CALL(cudaMemcpy(&arg.integral, arg.d_sum, sizeof(Real), cudaMemcpyDeviceToHost));
	arg.integral = 1.0 / sqrt(arg.integral * dx * dy); 

	CUDA_SAFE_CALL(cudaMemcpy(&maxvar, arg.d_max, sizeof(complex), cudaMemcpyDeviceToHost)); 
	CUDA_SAFE_CALL(cudaMemcpy(&minvar, arg.d_min, sizeof(complex), cudaMemcpyDeviceToHost)); 
	integral = arg.integral;
	//std::cout << "integral1=" << arg.integral << std::endl;
}






template <class Real>
void Normalize<Real>::callKernel(uint3 grid, uint3 block, size_t smem){ 
	switch(typ){
		case 0:
			callKernel_gm(grid, block, smem);
		break;
		case 1:
			callKernel_tex(grid, block, smem);
		break;
		default:
			callKernel_gm(grid, block, smem);
		break;
	}
}



template <class Real>
void Normalize<Real>::callKernel_gm(uint3 grid, uint3 block, size_t smem){
	Normalize_GetMaxMin<Real, true, false><<<grid, block, smem, 0>>>(arg);
}
template <class Real>
void Normalize<Real>::callKernel_tex(uint3 grid, uint3 block, size_t smem){
	Normalize_GetMaxMin<Real, true, true><<<grid, block, smem, 0>>>(arg);
}
		
template <class Real>
float Normalize<Real>::flops(float time_ms) const{
	return flop() * 1.0e-6 / time_ms;	
}
template <class Real>
long long Normalize<Real>::flop() const{
	return (6 * (nx-2) * (ny-2) + 2 * (nx*ny -1));	
}
template <class Real>
long long Normalize<Real>::bytes() const{
	return 2 * (nx-2) * (ny-2) * sizeof(complex);	
}
template <class Real>
float Normalize<Real>::bwdth(float time_ms) const{
	return bytes() * 1.0e-6 / time_ms;	
}

template class Normalize<float>;
template class Normalize<double>;






