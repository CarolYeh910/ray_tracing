#ifndef MATH_H
#define MATH_H

__device__ inline float fmin(float& a, float& b)
{
    return a <= b ? a : b;
}

__device__ inline float fmax(float& a, float& b)
{
    return a >= b ? a : b;
}

__device__ inline float fabs(float& a)
{
    return a >= 0 ? a : -a;
}

#endif
