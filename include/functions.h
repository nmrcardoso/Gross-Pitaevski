#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__


#include <cuda.h>
#include <thrust/device_vector.h>
#include "cuda_common.h"
#include <cuda_runtime_api.h>
#include "complex.h"


#include "structs.h"
#include "constants.h"


template<class Real>
inline __device__ Real gama(){
    Real res;
    return res;
}
template<>
inline __device__ float gama<float>(){
    return gammas;
}
template<>
inline __device__ double gama<double>(){
    return gammad;
}
template<class Real>
inline __device__ Real omega(){
    Real res;
    return res;
}
template<>
inline __device__ float omega<float>(){
    return omegas;
}
template<>
inline __device__ double omega<double>(){
    return omegad;
}

template<class Real>
inline __device__ Real g(){
    Real res;
    return res;
}
template<>
inline __device__ float g<float>(){
    return gs;
}
template<>
inline __device__ double g<double>(){
    return gd;
}


template<class Real>
inline __device__ Real dx(){}
template<>
inline __device__ float dx<float>(){ return dxs;}
template<>
inline __device__ double dx<double>(){ return dxd;}

template<class Real>
inline __device__ Real dy(){}
template<>
inline __device__ float dy<float>(){ return dys;}
template<>
inline __device__ double dy<double>(){ return dyd;}



template<class Real>
inline __device__ Real dx2(){}
template<>
inline __device__ float dx2<float>(){return dxs2;}
template<>
inline __device__ double dx2<double>(){return dxd2;}

template<class Real>
inline __device__ Real dy2(){}
template<>
inline __device__ float dy2<float>(){return dys2;}
template<>
inline __device__ double dy2<double>(){return dyd2;}




template<class Real>
inline __device__ Real oneoverdx(){ }
template<>
inline __device__ float oneoverdx<float>(){ return oneoverdxs; }
template<>
inline __device__ double oneoverdx<double>(){ return oneoverdxd; }


template<class Real>
inline __device__ Real oneoverdx2(){ }
template<>
inline __device__ float oneoverdx2<float>(){ return oneoverdx2s; }
template<>
inline __device__ double oneoverdx2<double>(){ return oneoverdx2d; }



template<class Real>
inline __device__ Real oneoverdy(){ }
template<>
inline __device__ float oneoverdy<float>(){ return oneoverdys; }
template<>
inline __device__ double oneoverdy<double>(){ return oneoverdyd; }


template<class Real>
inline __device__ Real oneoverdy2(){ }
template<>
inline __device__ float oneoverdy2<float>(){ return oneoverdy2s; }
template<>
inline __device__ double oneoverdy2<double>(){ return oneoverdy2d; }









template<class Real>
inline __device__ Real Lx(){ }
template<>
inline __device__ float Lx<float>(){ return Lxs; }
template<>
inline __device__ double Lx<double>(){ return Lxd; }




template<class Real>
inline __device__ Real Ly(){ }
template<>
inline __device__ float Ly<float>(){ return Lys; }
template<>
inline __device__ double Ly<double>(){ return Lyd; }























template<class Real>
inline __device__ Real dt(){
    Real res;
    return res;
}
template<>
inline __device__ float dt<float>(){
    return dts;
}
template<>
inline __device__ double dt<double>(){
    return dtd;
}







template<class Real>
inline __device__ complex InvIminusGamma(){
    complex res;
    return res;
}
template<>
inline __device__ complexs InvIminusGamma<float>(){
    return invIminusGammas;
}
template<>
inline __device__ complexd InvIminusGamma<double>(){
    return invIminusGammad;
}

template<class Real>
inline __device__ complex Iomega(){
    complex res;
    return res;
}
template<>
inline __device__ complexs Iomega<float>(){
    return Iomegas;
}
template<>
inline __device__ complexd Iomega<double>(){
    return Iomegad;
}



template<class Real>
struct SharedMemory
{
    __device__ inline operator       Real*()
    {
        extern volatile __shared__ Real __shmem[];
        return (Real*)__shmem;
    }

    __device__ inline operator const Real*() const
    {
        extern volatile __shared__ Real __shmem[];
        return (Real*)__shmem;
    }
};









template<>
struct SharedMemory<float>
{
    __device__ inline operator       float*()
    {
        extern volatile __shared__ float __shmemfs[];
        return (float*)__shmemfs;
    }

    __device__ inline operator const float*() const
    {
        extern volatile __shared__ float __shmemfs[];
        return (float*)__shmemfs;
    }
};
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern volatile __shared__ double __shmemdd[];
        return (double*)__shmemdd;
    }

    __device__ inline operator const double*() const
    {
        extern volatile __shared__ double __shmemdd[];
        return (double*)__shmemdd;
    }
};





template<>
struct SharedMemory<complexs>
{
    __device__ inline operator       complexs*()
    {
        extern volatile __shared__ complexs __shmems[];
        return (complexs*)__shmems;
    }

    __device__ inline operator const complexs*() const
    {
        extern volatile __shared__ complexs __shmems[];
        return (complexs*)__shmems;
    }
};
template<>
struct SharedMemory<complexd>
{
    __device__ inline operator       complexd*()
    {
        extern volatile __shared__ complexd __shmemd[];
        return (complexd*)__shmemd;
    }

    __device__ inline operator const complexd*() const
    {
        extern volatile __shared__ complexd __shmemd[];
        return (complexd*)__shmemd;
    }
};


__device__ inline int index(int i, int j){

    return threadIdx.x + 1 + i + (threadIdx.y + 1 + j) * (blockDim.x+2);
}






// ================================================
// Templates for handling "float" and "double" data
// ================================================

// The default implementation for atomic maximum
template <typename T>
__inline__ __device__ void CudaAtomicMax(T * const address, const T value)
{
	atomicMax(address, value);
}

/**
 * @brief Compute the maximum of 2 single-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
template <>
__inline__ __device__ void CudaAtomicMax(float * const address, const float value)
{
	if (* address >= value)
	{
		return;
	}

	int * const address_as_i = (int *)address;
	int old = * address_as_i, assumed;

	do 
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}

/**
 * @brief Compute the maximum of 2 double-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
template <>
__inline__ __device__ void CudaAtomicMax(double * const address, const double value)
{
	if (* address >= value)
	{
		return;
	}

	unsigned long long int*  address_as_i = (unsigned long long int *)address;
    unsigned long long int old = * address_as_i, assumed;

	do 
	{
        assumed = old;
		if (__longlong_as_double(assumed) >= value)
		{
			break;
		}
		
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}


template <>
__inline__ __device__ void CudaAtomicMax(complexd *addr, complexd val){
    CudaAtomicMax((double*)addr, val.real());
    CudaAtomicMax((double*)addr+1, val.imag());
  }

template <>
__inline__ __device__ void CudaAtomicMax(complexs *addr, complexs val){
    CudaAtomicMax((float*)addr, val.real());
    CudaAtomicMax((float*)addr+1, val.imag());
  }










template <typename T>
__inline__ __device__ void CudaAtomicMin(T * const address, const T value)
{
	atomicMin(address, value);
}

template <>
__inline__ __device__ void CudaAtomicMin(float * const address, const float value)
{
	if (* address <= value)
	{
		return;
	}

	int * const address_as_i = (int *)address;
	int old = * address_as_i, assumed;

	do 
	{
		assumed = old;
		if (__int_as_float(assumed) <= value)
		{
			break;
		}

		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}


template <>
__inline__ __device__ void CudaAtomicMin(double * const address, const double value)
{
	if (* address <= value)
	{
		return;
	}

	unsigned long long int*  address_as_i = (unsigned long long int *)address;
    unsigned long long int old = * address_as_i, assumed;

	do 
	{
        assumed = old;
		if (__longlong_as_double(assumed) <= value)
		{
			break;
		}
		
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}


template <>
__inline__ __device__ void CudaAtomicMin(complexd *addr, complexd val){
    CudaAtomicMin((double*)addr, val.real());
    CudaAtomicMin((double*)addr+1, val.imag());
  }

template <>
__inline__ __device__ void CudaAtomicMin(complexs *addr, complexs val){
    CudaAtomicMin((float*)addr, val.real());
    CudaAtomicMin((float*)addr+1, val.imag());
  }






__inline__ __device__ float CudaAtomicAdd(float *addr, float val){
    return atomicAdd(addr, val);
  }


__inline__ __device__ double CudaAtomicAdd(double *addr, double val){
  double old=*addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
            __double_as_longlong(assumed),
            __double_as_longlong(val+assumed)));
  } while( __double_as_longlong(assumed)!=__double_as_longlong(old) );
  
  return old;
}

__inline__ __device__ complexd CudaAtomicAdd(complexd *addr, complexd val){
    complexd old=*addr;
    old.real() = CudaAtomicAdd((double*)addr, val.real());
    old.imag() = CudaAtomicAdd((double*)addr+1, val.imag());
    return old;
  }

__inline__ __device__ complexs CudaAtomicAdd(complexs *addr, complexs val){
    complexs old=*addr;
    old.real() = CudaAtomicAdd((float*)addr, val.real());
    old.imag() = CudaAtomicAdd((float*)addr+1, val.imag());
    return old;
  }
















#endif

