#include <Python.h>
#include "emc_cuda.h"

using namespace std;

const int NTHREADS = 256;
const int RESPONSABILITY_THRESHOLD = 1e-10f;

#define cudaErrorCheck(ans) {_cudaErrorCheck((ans), __FILE__, __LINE__);}
inline void _cudaErrorCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "cudaErrorCheck: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}


template<typename T>
__device__ void inblock_reduce(T * data){
  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1){
    if (threadIdx.x < s){
      data[threadIdx.x] += data[threadIdx.x + s];
    }
    __syncthreads();
  }  
}

float *int_to_float_pointer(const unsigned long long pointer_int)
{
  float *pointer = (float *)pointer_int;
  return pointer;
}

int *int_to_int_pointer(const unsigned long long pointer_int)
{
  int *pointer = (int *)pointer_int;
  return pointer;
}

__global__ void add_one_kernel(float *array, const int length)
{
  int index = blockDim.x*blockIdx.x + threadIdx.x;
  if (index < length) {
    array[index] += 1.;
  }
}

//void cuda_add_one(float *const array, const int length)
void cuda_add_one(float *const array, const int length)
{
  const int nthreads = 256;
  const int nblocks = (length+nthreads-1)/nthreads;
  add_one_kernel<<<nblocks, nthreads>>>(array, length);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__device__ void device_interpolate_get_coordinate_weight(const float coordinate, const int side,
							 int *low_coordinate, float *low_weight, int *out_of_range)
{
  if (coordinate > -0.5 && coordinate <= 0.) {
    *low_weight = 0.;
    *low_coordinate = -1;
  } else if (coordinate > 0. && coordinate <= (side-1)) {
    *low_weight = ceil(coordinate) - coordinate;
    *low_coordinate = (int)ceil(coordinate) - 1;
  } else if (coordinate > (side-1) && coordinate < (side-0.5)) {
    *low_weight = 1.;
    *low_coordinate = side-1;
  } else {
    *out_of_range = 1;
  }
}

__device__ float device_model_get(const float *const model, const int model_x, const int model_y, const int model_z,
				  const float coordinate_x, const float coordinate_y, const float coordinate_z)
{
  int low_x, low_y, low_z;
  float low_weight_x, low_weight_y, low_weight_z;
  int out_of_range = 0;
  device_interpolate_get_coordinate_weight(coordinate_x, model_x, &low_x, &low_weight_x, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_y, model_y, &low_y, &low_weight_y, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_z, model_z, &low_z, &low_weight_z, &out_of_range);

  if (out_of_range != 0) {
    return -1.f;
  } else {
    float interp_sum = 0.;
    float interp_weight = 0.;
    int index_x, index_y, index_z;
    float weight_x, weight_y, weight_z;
    for (index_x = low_x; index_x <= low_x+1; index_x += 1) {
      if (index_x == low_x && low_weight_x == 0.) continue;
      if (index_x == (low_x+1) && low_weight_x == 1.) continue;
      if (index_x == low_x) weight_x = low_weight_x;
      else weight_x = 1. - low_weight_x;

      for (index_y = low_y; index_y <= low_y+1; index_y += 1) {
	if (index_y == low_y && low_weight_y == 0.) continue;
	if (index_y == (low_y+1) && low_weight_y == 1.) continue;
	if (index_y == low_y) weight_y = low_weight_y;
	else weight_y = 1. - low_weight_y;

	for (index_z = low_z; index_z <= low_z+1; index_z += 1) {
	  if (index_z == low_z && low_weight_z == 0.) continue;
	  if (index_z == (low_z+1) && low_weight_z == 1.) continue;
	  if (index_z == low_z) weight_z = low_weight_z;
	  else weight_z = 1. - low_weight_z;

	  if (model[model_x*model_y*index_z + model_x*index_y + index_x] >= 0.) {
	    interp_sum += weight_x*weight_y*weight_z*model[model_x*model_y*index_z + model_x*index_y + index_x];
	    interp_weight += weight_x*weight_y*weight_z;
	  }
	}
      }
    }
    if (interp_weight > 0.) {
      return interp_sum / interp_weight;
    } else {
      return -1.f;
    }
  }
}

__device__ void device_get_slice(const float *const model, const int model_x, const int model_y, const int model_z,
				 float *const slice, const int image_x, const int image_y,
				 const float *const rotation,
				 const float *const coordinates) {  
  const float *const coordinates_0 = &coordinates[0*image_x*image_y];
  const float *const coordinates_1 = &coordinates[1*image_x*image_y];
  const float *const coordinates_2 = &coordinates[2*image_x*image_y];

  float m00 = rotation[0]*rotation[0] + rotation[1]*rotation[1] - rotation[2]*rotation[2] - rotation[3]*rotation[3];
  float m01 = 2.0f*rotation[1]*rotation[2] - 2.0f*rotation[0]*rotation[3];
  float m02 = 2.0f*rotation[1]*rotation[3] + 2.0f*rotation[0]*rotation[2];
  float m10 = 2.0f*rotation[1]*rotation[2] + 2.0f*rotation[0]*rotation[3];
  float m11 = rotation[0]*rotation[0] - rotation[1]*rotation[1] + rotation[2]*rotation[2] - rotation[3]*rotation[3];
  float m12 = 2.0f*rotation[2]*rotation[3] - 2.0f*rotation[0]*rotation[1];
  float m20 = 2.0f*rotation[1]*rotation[3] - 2.0f*rotation[0]*rotation[2];
  float m21 = 2.0f*rotation[2]*rotation[3] + 2.0f*rotation[0]*rotation[1];
  float m22 = rotation[0]*rotation[0] - rotation[1]*rotation[1] - rotation[2]*rotation[2] + rotation[3]*rotation[3];

  float new_x, new_y, new_z;
  for (int x = 0; x < image_x; x++) {
    for (int y = threadIdx.x; y < image_y; y+=blockDim.x) {
      /* This is just a matrix multiplication with rotation */
      new_x = m00*coordinates_0[y*image_x+x] + m01*coordinates_1[y*image_x+x] + m02*coordinates_2[y*image_x+x] + model_x/2.0 - 0.5;
      new_y = m10*coordinates_0[y*image_x+x] + m11*coordinates_1[y*image_x+x] + m12*coordinates_2[y*image_x+x] + model_y/2.0 - 0.5;
      new_z = m20*coordinates_0[y*image_x+x] + m21*coordinates_1[y*image_x+x] + m22*coordinates_2[y*image_x+x] + model_z/2.0 - 0.5;

      slice[y*image_x+x] = device_model_get(model, model_x, model_y, model_z, new_x, new_y, new_z);
    }
  }
}

__global__ void kernel_expand_model(const float *const model, const int model_x, const int model_y, const int model_z,
				    float *const slices, const int image_x, const int image_y,
				    const float *const rotations, const float *const coordinates)
{
  const int rotation_index = blockIdx.x;
  device_get_slice(model, model_x, model_y, model_z,
		   &slices[image_x*image_y*rotation_index], image_x, image_y,
		   &rotations[4*rotation_index], coordinates);
}
				    

void cuda_expand_model(const float *const model, const int model_x, const int model_y, const int model_z,
		       float *const slices, const int image_x, const int image_y,
		       const float *const rotations, const int number_of_rotations,
		       const float *const coordinates)
{
  int nblocks = number_of_rotations;
  int nthreads = NTHREADS;
  kernel_expand_model<<<nblocks, nthreads>>>(model, model_x, model_y, model_z, slices, image_x, image_y, rotations, coordinates);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__device__ void device_model_set(float *const model, float *const model_weights,
			    const int model_x, const int model_y, const int model_z,
			    const float coordinate_x, const float coordinate_y, const float coordinate_z,
			    const float value, const float value_weight)
{
  int low_x, low_y, low_z;
  float low_weight_x, low_weight_y, low_weight_z;
  int out_of_range = 0;
  device_interpolate_get_coordinate_weight(coordinate_x, model_x, &low_x, &low_weight_x, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_y, model_y, &low_y, &low_weight_y, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_z, model_z, &low_z, &low_weight_z, &out_of_range);

  if (out_of_range == 0) {
    int index_x, index_y, index_z;
    float weight_x, weight_y, weight_z;
    for (index_x = low_x; index_x <= low_x+1; index_x += 1) {
      if (index_x == low_x && low_weight_x == 0.) continue;
      if (index_x == (low_x+1) && low_weight_x == 1.) continue;
      if (index_x == low_x) weight_x = low_weight_x;
      else weight_x = 1. - low_weight_x;

      for (index_y = low_y; index_y <= low_y+1; index_y += 1) {
	if (index_y == low_y && low_weight_y == 0.) continue;
	if (index_y == (low_y+1) && low_weight_y == 1.) continue;
	if (index_y == low_y) weight_y = low_weight_y;
	else weight_y = 1. - low_weight_y;

	for (index_z = low_z; index_z <= low_z+1; index_z += 1) {
	  if (index_z == low_z && low_weight_z == 0.) continue;
	  if (index_z == (low_z+1) && low_weight_z == 1.) continue;
	  if (index_z == low_z) weight_z = low_weight_z;
	  else weight_z = 1. - low_weight_z;
	  
	  atomicAdd(&model[model_x*model_y*index_z + model_x*index_y + index_x],
		    weight_x*weight_y*weight_z*value_weight*value);
	  atomicAdd(&model_weights[model_x*model_y*index_z + model_x*index_y + index_x],
		    weight_x*weight_y*weight_z*value_weight);
	}
      }
    }
  }
}

__device__ void device_insert_slice(float *const model, float *const model_weights,
				    const int model_x, const int model_y, const int model_z,
				    const float *const slice, const int image_x, const int image_y,
				    const float slice_weight,
				    const float *const rotation, const float *const coordinates)
{
  const float *const coordinates_0 = &coordinates[0*image_x*image_y];
  const float *const coordinates_1 = &coordinates[1*image_x*image_y];
  const float *const coordinates_2 = &coordinates[2*image_x*image_y];

  float m00 = rotation[0]*rotation[0] + rotation[1]*rotation[1] - rotation[2]*rotation[2] - rotation[3]*rotation[3];
  float m01 = 2.0f*rotation[1]*rotation[2] - 2.0f*rotation[0]*rotation[3];
  float m02 = 2.0f*rotation[1]*rotation[3] + 2.0f*rotation[0]*rotation[2];
  float m10 = 2.0f*rotation[1]*rotation[2] + 2.0f*rotation[0]*rotation[3];
  float m11 = rotation[0]*rotation[0] - rotation[1]*rotation[1] + rotation[2]*rotation[2] - rotation[3]*rotation[3];
  float m12 = 2.0f*rotation[2]*rotation[3] - 2.0f*rotation[0]*rotation[1];
  float m20 = 2.0f*rotation[1]*rotation[3] - 2.0f*rotation[0]*rotation[2];
  float m21 = 2.0f*rotation[2]*rotation[3] + 2.0f*rotation[0]*rotation[1];
  float m22 = rotation[0]*rotation[0] - rotation[1]*rotation[1] - rotation[2]*rotation[2] + rotation[3]*rotation[3];

  float new_x, new_y, new_z;
  for (int x = 0; x < image_x; x++) {
    for (int y = threadIdx.x; y < image_y; y+=blockDim.x) {
      /* This is just a matrix multiplication with rotation */
      new_x = m00*coordinates_0[y*image_x+x] + m01*coordinates_1[y*image_x+x] + m02*coordinates_2[y*image_x+x] + model_x/2.0 - 0.5;
      new_y = m10*coordinates_0[y*image_x+x] + m11*coordinates_1[y*image_x+x] + m12*coordinates_2[y*image_x+x] + model_y/2.0 - 0.5;
      new_z = m20*coordinates_0[y*image_x+x] + m21*coordinates_1[y*image_x+x] + m22*coordinates_2[y*image_x+x] + model_z/2.0 - 0.5;

      device_model_set(model, model_weights, model_x, model_y, model_z, new_x, new_y, new_z, slice[y*image_x+x], slice_weight);
    }
  }
}


__global__ void kernel_insert_slices(float *const model, float *const model_weights,
				     const int model_x, const int model_y, const int model_z,
				     const float *const slices, const int image_x, const int image_y,
				     const float *const slice_weights, const float *const rotations,
				     const float *const coordinates) {
  const int rotation_index = blockIdx.x;
  device_insert_slice(model, model_weights, model_x, model_y, model_z,
		      &slices[image_x*image_y*rotation_index], image_x, image_y,
		      slice_weights[rotation_index], &rotations[4*rotation_index],
		      coordinates);
}

void cuda_insert_slices(float *const model, float *const model_weights,
			const int model_x, const int model_y, const int model_z,
			const float *const slices, const int image_x, const int image_y,
			const float *const slice_weights,
			const float *const rotations, const int number_of_rotations,
			const float *const coordinates)
{
  int nblocks = number_of_rotations;
  int nthreads = NTHREADS;
  kernel_insert_slices<<<nblocks, nthreads>>>(model, model_weights, model_x, model_y, model_z,
					      slices, image_x, image_y, slice_weights, rotations,
					      coordinates);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());

}

__global__ void kernel_update_slices(float *const slices, const float *const patterns, const int number_of_patterns, const int number_of_pixels,
				     const float *const responsabilities)
{
  const int rotation_index = blockIdx.x;
  for (int pixel_index = threadIdx.x; pixel_index < number_of_pixels; pixel_index += blockDim.x) {
    float sum = 0.;
    float weight = 0.;
    for (int pattern_index = 0; pattern_index < number_of_patterns; pattern_index++) {
      sum += patterns[pattern_index*number_of_pixels + pixel_index] * responsabilities[rotation_index*number_of_patterns + pattern_index];
      weight += responsabilities[rotation_index*number_of_patterns + pattern_index];
    }
    if (weight > 0.) {
      slices[rotation_index*number_of_pixels + pixel_index] = sum / weight;
    } else {
      slices[rotation_index*number_of_pixels + pixel_index] = -1.;
    }
  }
}

void cuda_update_slices(float *const slices, const int number_of_rotations, const float *const patterns, const int number_of_patterns,
			const int image_x, const int image_y, const float *const responsabilities)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices<<<nblocks, nthreads>>>(slices, patterns, number_of_patterns, image_x*image_y, responsabilities);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_responsabilities(const float* const patterns, const float *const slices, const int number_of_pixels, float *const responsabilities, const float sigma)
{
  __shared__ float sum_cache[NTHREADS];
  __shared__ float weight_cache[NTHREADS];

  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;
  
  const float *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];
  
  /* Use a gaussian with a sqrt normalization */
  float sum = 0.;
  float weight = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index+= blockDim.x) {
    if (pattern[index] >= 0. && slice[index] > 0.) {
      sum += pow((slice[index] - pattern[index]) / sqrt(slice[index]), 2);
      weight += 1.;
    }
  }
  sum_cache[threadIdx.x] = sum;
  weight_cache[threadIdx.x] = weight;

  inblock_reduce(sum_cache);
  inblock_reduce(weight_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -sum_cache[0]/(2.*weight_cache[0]*pow(sigma, 2));
  }
}

void cuda_calculate_responsabilities(const float *const patterns, const int number_of_patterns, const float *const slices, const int number_of_rotations,
				     const int image_x, const int image_y, float *const responsabilities, const float sigma)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_responsabilities<<<nblocks, nthreads>>>(patterns, slices, image_x*image_y, responsabilities, sigma);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

//void cuda_update_slices(



