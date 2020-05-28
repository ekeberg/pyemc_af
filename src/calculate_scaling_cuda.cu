#include <Python.h>
#include <emc_cuda.h>	
#include "cuda_tools.h"

using namespace std;

__global__ void kernel_calculate_scaling_poisson(const float *const patterns,
						 const float *const slices,
						 float *const scaling,
						 const int number_of_pixels) {
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;

  const float *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];

  float sum_slice = 0.;
  float sum_pattern = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (pattern[index] >= 0. && slice[index] >= 0.) {
      sum_slice += slice[index];
      sum_pattern += pattern[index];
    }
  }

  __shared__ float sum_slice_cache[NTHREADS];
  __shared__ float sum_pattern_cache[NTHREADS];  
  sum_slice_cache[threadIdx.x] = sum_slice;
  sum_pattern_cache[threadIdx.x] = sum_pattern;
  inblock_reduce(sum_slice_cache);
  inblock_reduce(sum_pattern_cache);

  if (threadIdx.x == 0) {
    scaling[index_slice*number_of_patterns + index_pattern] = sum_slice_cache[0] / sum_pattern_cache[0];
  }
}

void calculate_scaling_poisson(const float *const patterns,
			       const int number_of_patterns,
			       const float *const slices,
			       const int number_of_rotations,
			       const int number_of_pixels,
			       float *const scaling)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_scaling_poisson<<<nblocks, nthreads>>>(patterns,
							  slices,
							  scaling,
							  number_of_pixels);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_scaling_poisson_sparse(const int *const pattern_start_indices,
							const int *const pattern_indices,
							const float *const pattern_values,
							const float *const slices,
							float *const scaling,
							const int number_of_pixels) {
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;

  //const float *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];

  const int this_start_index = pattern_start_indices[index_pattern];
  const int this_end_index = pattern_start_indices[index_pattern+1];

  float sum_slice = 0.;
  float sum_pattern = 0.;

  for (int index = this_start_index+threadIdx.x; index < this_end_index; index += blockDim.x) {
    if (slice[pattern_indices[index]]) {
      sum_pattern += pattern_values[index];
    }
  }

  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (slice[index] >= 0.) {
      sum_slice += slice[index];
    }
  }

  __shared__ float sum_slice_cache[NTHREADS];
  __shared__ float sum_pattern_cache[NTHREADS];  
  sum_slice_cache[threadIdx.x] = sum_slice;
  sum_pattern_cache[threadIdx.x] = sum_pattern;
  inblock_reduce(sum_slice_cache);
  inblock_reduce(sum_pattern_cache);

  if (threadIdx.x == 0) {
    scaling[index_slice*number_of_patterns + index_pattern] = sum_slice_cache[0] / sum_pattern_cache[0];
  }
}

void calculate_scaling_poisson_sparse(const int *const pattern_start_indices,
				      const int *const pattern_indices,
				      const float *const pattern_values,
				      const int number_of_patterns,
				      const float *const slices,
				      const int number_of_rotations,
				      const int number_of_pixels,
				      float *const scaling)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_scaling_poisson_sparse<<<nblocks, nthreads>>>(pattern_start_indices,
								 pattern_indices,
								 pattern_values,
								 slices,
								 scaling,
								 number_of_pixels);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}
