#ifndef __INIT_H__
#define __INIT_H__

#include <iostream>


#include <cuda.h>
#include "complex.h"
#include "timer.h"
#include <thrust/device_vector.h>




template<class Real>
void Init(complex *a0_d, Real* d_sum, int nx, int ny, bool randomInit);


#endif
