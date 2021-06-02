#ifndef CUDA_VECTOR_TYPES_H
#define CUDA_VECTOR_TYPES_H

#include <vector_functions.h>
#include <vector_types.h>



template <typename Real, int number> struct MakeVector;
template <> struct MakeVector<float, 2>
{
    typedef float2 type;
};
template <> struct MakeVector<double, 2>
{
    typedef double2 type;
};
template <> struct MakeVector<float, 4>
{
    typedef float4 type;
};
template <> struct MakeVector<double, 4>
{
    typedef double4 type;
};


#endif // #ifndef CUDA_VECTOR_TYPES_H
