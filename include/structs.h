#ifndef __STRUCTS_H__
#define __STRUCTS_H__



#include "complex.h"





template<typename T>
struct MaxVal
{
 __host__ __device__ __forceinline__
  T operator()(const T& a, const T& b) {
	if(  a > b ) return a;
    return b;
  }
};





template<>
struct MaxVal<complexs> {
 __host__ __device__ __forceinline__
  complexs operator()(const complexs& a, const complexs& b) {
	complexs tmp = b;
	if(  a.val.x > b.val.x ) tmp.val.x = a.val.x;
	if(  a.val.y > b.val.y ) tmp.val.y = a.val.y;
	return tmp;
  }
};


template<>
struct MaxVal<complexd> {
 __host__ __device__ __forceinline__
  complexd operator()(const complexd& a, const complexd& b) {
	complexd tmp = b;
	if(  a.val.x > b.val.x ) tmp.val.x = a.val.x;
	if(  a.val.y > b.val.y ) tmp.val.y = a.val.y;
	return tmp;
  }
};




template<typename T>
struct MinVal
{
 __host__ __device__ __forceinline__
  T operator()(const T& a, const T& b) {
	if(  a < b ) return a;
    return b;
  }
};





template<>
struct MinVal<complexs> {
 __host__ __device__ __forceinline__
  complexs operator()(const complexs& a, const complexs& b) {
	complexs tmp = b;
	if(  a.val.x < b.val.x ) tmp.val.x = a.val.x;
	if(  a.val.y < b.val.y ) tmp.val.y = a.val.y;
	return tmp;
  }
};


template<>
struct MinVal<complexd> {
 __host__ __device__ __forceinline__
  complexd operator()(const complexd& a, const complexd& b) {
	complexd tmp = b;
	if(  a.val.x < b.val.x ) tmp.val.x = a.val.x;
	if(  a.val.y < b.val.y ) tmp.val.y = a.val.y;
	return tmp;
  }
};





template<typename T>
struct Summ
{
 __host__ __device__ __forceinline__
  T operator()(const T& a, const T& b) {
	return a+b;
  }
};






#endif

