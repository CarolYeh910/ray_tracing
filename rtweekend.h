#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <limits>
#include <curand_kernel.h>

// Usings

// Constants

__device__ float infinity = std::numeric_limits<float>::infinity();
__device__ float pi = 3.1415926535897932385f;

// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__global__ void random_init(int image_width, int image_height, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= image_width) || (j >= image_height)) return;
	int pixel_index = j * image_width + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ inline float random_float(curandState* local_rand_state) {
    // Returns a random real in [0,1).
    return curand_uniform(local_rand_state);
}

__device__ inline float random_float(float min, float max, curandState* local_rand_state) {
    // Returns a random real in [min,max).
	return min + (max - min) * random_float(local_rand_state);
}

__device__ inline int random_int(int min, int max, curandState* local_rand_state) {
    // Returns a random integer in [min,max].
	return static_cast<int>(random_float(min, max + 1, local_rand_state));
}

__device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif