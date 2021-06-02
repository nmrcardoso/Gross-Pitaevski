#ifndef __TEXTURES_H__
#define __TEXTURES_H__


#include <cuda.h>
#include <thrust/device_vector.h>
#include "cuda_common.h"
#include <cuda_runtime_api.h>
#include "complex.h"
#include "constants.h"



extern texture<float2, 1, cudaReadModeElementType> texsp0;
extern texture<int4, 1, cudaReadModeElementType> texdp0;
extern texture<float2, 1, cudaReadModeElementType> texsp1;
extern texture<int4, 1, cudaReadModeElementType> texdp1;
extern texture<float2, 1, cudaReadModeElementType> texsp2;
extern texture<int4, 1, cudaReadModeElementType> texdp2;



template <class Real, int typ> 
inline __device__ complex TEXTURE(uint id){
	return complex::zero();
}
template <>
inline __device__ complexs TEXTURE<float, 0>(uint id){
	return make_complexs(tex1Dfetch(texsp0, id));
}
template <>
inline __device__ complexs TEXTURE<float, 1>(uint id){
	return make_complexs(tex1Dfetch(texsp1, id));
}
template <>
inline __device__ complexs TEXTURE<float, 2>(uint id){
	return make_complexs(tex1Dfetch(texsp2, id));
}


template <>
inline __device__ complexd TEXTURE<double, 0>(uint id){
    int4 u = tex1Dfetch(texdp0, id);
    return  make_complexd(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}
template <>
inline __device__ complexd TEXTURE<double, 1>(uint id){
    int4 u = tex1Dfetch(texdp1, id);
    return  make_complexd(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}
template <>
inline __device__ complexd TEXTURE<double, 2>(uint id){
    int4 u = tex1Dfetch(texdp2, id);
    return  make_complexd(__hiloint2double(u.y, u.x), __hiloint2double(u.w, u.z));
}





template <bool UseTex, int typ, class Real> 
__device__ inline complex ELEM_LOAD(const complex *array, const uint id){
    if (UseTex) return TEXTURE<Real, typ>( id);
    else return array[id];
}
























__inline__ __device__ complexs getdata0(complexs *x, int i){
if(UseTex)
  return make_complexs(tex1Dfetch(texsp0, i));
else 
  return x[i];
}

__inline__ __device__ complexd getdata0(complexd *x, int i){
if(UseTex){
// double requires Compute Capability 1.3 or greater
#if __CUDA_ARCH__ >= 130
  int4 v = tex1Dfetch(texdp0, i);
  return make_complexd(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
#else
  return x[i];
#endif
}
else 
  return x[i];
}
__inline__ __device__ complexs getdata1(complexs *x, int i){
if(UseTex)
  return make_complexs(tex1Dfetch(texsp1, i));
else 
  return x[i];
}

__inline__ __device__ complexd getdata1(complexd *x, int i){
if(UseTex){
// double requires Compute Capability 1.3 or greater
#if __CUDA_ARCH__ >= 130
  int4 v = tex1Dfetch(texdp1, i);
  return make_complexd(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
#else
  return x[i];
#endif
}
else 
  return x[i];
}
__inline__ __device__ complexs getdata2(complexs *x, int i){
if(UseTex)
  return make_complexs(tex1Dfetch(texsp2, i));
else 
  return x[i];
}

__inline__ __device__ complexd getdata2(complexd *x, int i){
if(UseTex){
// double requires Compute Capability 1.3 or greater
#if __CUDA_ARCH__ >= 130
  int4 v = tex1Dfetch(texdp2, i);
  return make_complexd(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
#else
  return x[i];
#endif
}
else 
  return x[i];
}
#endif

