#include <iostream>

#include <SDL2/SDL.h>

#include <string>
#include <stdexcept>
#include <cstdio>


#include <cuda.h>
#include <curand.h>
#include "cuda_common.h"
#include <cuda_runtime_api.h>



#include "complex.h"
#include "cuda_common.h"
#include "constants.h"


//https://nvlabs.github.io/cub/
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include <typeinfo>


#include "textures.h"
#include "functions.h"




#define CURAND_CALL(x) x

//do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      return EXIT_FAILURE;}} while(0)



template<class Real>
__global__ void initializeRI(complex *in, Real *d_sum){

	uint id = threadIdx.x + blockIdx.x * blockDim.x;
	uint i = id % nx;
	uint j = id / nx;


	Real val = 0.0;
	if(i < nx && j < ny){
		if(i>0 && j>0 && i<nx-1 && j<ny-1) val = in[i+j*nx].abs2();
		else in[i+j*nx] = complex::zero();
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
		CudaAtomicAdd(d_sum, sum);
	}
}


template<class Real>
__global__ void initialize(complex *in, Real *d_sum){

	uint id = threadIdx.x + blockIdx.x * blockDim.x;
	uint i = id % nx;
	uint j = id / nx;


	Real val = 0.0;
	if(i>0 && j>0 && i<nx-1 && j<ny-1){
		complex value = complex::one();
		in[i+j*nx]= value;
		val = value.abs2();
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
		CudaAtomicAdd(d_sum, sum);
	}
}



template<class Real>
void cGenerateUniform(curandGenerator_t gen, Real* a0, uint size){};



template<>
void cGenerateUniform<float>(curandGenerator_t gen, float* a0, uint size){
	CURAND_CALL(curandGenerateUniform(gen, a0, size));
}
template<>
void cGenerateUniform<double>(curandGenerator_t gen, double* a0, uint size){
	CURAND_CALL(curandGenerateUniformDouble(gen, a0, size));
}


template<class Real>
void Init(complex *a0_d, Real* d_sum, int nx, int ny, bool randomInit){

	CUDA_SAFE_CALL(cudaMemset(d_sum, 0, sizeof(Real))); 

    dim3 threadsperblock(128);
    dim3 nblocks((nx*ny+threadsperblock.x-1)/threadsperblock.x, 1, 1);
	size_t shared_bytes = threadsperblock.x * sizeof(Real);
    
	if(randomInit){
		curandGenerator_t gen; 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
		cGenerateUniform(gen, (Real*)a0_d, nx*ny*2);
    	CURAND_CALL(curandDestroyGenerator(gen));
		initializeRI<Real><<<nblocks,threadsperblock, shared_bytes, 0>>>(a0_d, d_sum);
	}
	else		initialize<Real><<<nblocks,threadsperblock, shared_bytes, 0>>>(a0_d, d_sum);
    CUT_CHECK_ERROR("kernel launch failure");
    CUDA_SAFE_THREAD_SYNC(); 
    
}

template
void Init<float>(complexs *a0_d, float* d_sum, int nx, int ny, bool randomInit);
template
void Init<double>(complexd *a0_d, double* d_sum, int nx, int ny, bool randomInit);




