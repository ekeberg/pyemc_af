#include <Python.h>
#include <emc_cuda.h>	
#include "cuda_tools.h"

using namespace std;

__global__ void kernel_calculate_responsabilities(const float* const patterns,
						  const float *const slices,
						  const int number_of_pixels,
						  float *const responsabilities,
						  const float sigma)
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
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
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

void calculate_responsabilities(const float *const patterns,
				const int number_of_patterns,
				const float *const slices,
				const int number_of_rotations,
				const int image_x,
				const int image_y,
				float *const responsabilities,
				const float sigma)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_responsabilities<<<nblocks, nthreads>>>(patterns,
							   slices,
							   image_x*image_y,
							   responsabilities,
							   sigma);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_responsabilities_poisson(const float* const patterns,
							  const float *const slices,
							  const int number_of_pixels,
							  float *const responsabilities,
							  const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;
  
  const float *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];
  
  /* Use a gaussian with a sqrt normalization */
  float sum = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (pattern[index] >= 0. && slice[index] > 0.) {
      sum += ((-slice[index]) +
	      ((int) pattern[index]) * logf(slice[index]) -
	      log_factorial_table[(int) pattern[index]]);
    }
  }
  sum_cache[threadIdx.x] = sum;

  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = sum_cache[0];
  }
}

void calculate_responsabilities_poisson(const float *const patterns,
					const int number_of_patterns,
					const float *const slices,
					const int number_of_rotations,
					const int image_x,
					const int image_y,
					float *const responsabilities,
					const float *const log_factorial_table)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_responsabilities_poisson<<<nblocks, nthreads>>>(patterns,
								   slices,
								   image_x*image_y,
								   responsabilities,
								   log_factorial_table);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_responsabilities_poisson_scaling(const float *const patterns,
								  const float *const slices,
								  const int number_of_pixels,
								  const float *const scalings,
								  float *const responsabilities,
								  const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;
  
  const float *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];
  const float scaling = scalings[index_slice*number_of_patterns + index_pattern];
  
  // Use a gaussian with a sqrt normalization
  float sum = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (pattern[index] >= 0. && slice[index] > 0.) {
      sum += ((-slice[index]/scaling) +
	      ((int) pattern[index]) * logf(slice[index]/scaling) -
	      log_factorial_table[(int) pattern[index]]);
    }
  }
  sum_cache[threadIdx.x] = sum;

  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = sum_cache[0];
  }
}

void calculate_responsabilities_poisson_scaling(const float *const patterns,
						const int number_of_patterns,
						const float *const slices,
						const int number_of_rotations,
						const int image_x,
						const int image_y,
						const float *const scalings,
						float *const responsabilities,
						const float *const log_factorial_table)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_responsabilities_poisson_scaling<<<nblocks, nthreads>>>(patterns,
									   slices,
									   image_x*image_y,
									   scalings,
									   responsabilities,
									   log_factorial_table);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_sum_slices(const float *const slices,
				  const int number_of_pixels,
				  float *const slice_sums)
{
  __shared__ float sum_cache[NTHREADS];
  
  const int index_slice = blockIdx.x;
  const float *const slice = &slices[number_of_pixels*index_slice];

  float sum = 0.;
  for (int index_pixel = threadIdx.x; index_pixel < number_of_pixels; index_pixel += blockDim.x) {
    if (slice[index_pixel] > 0.) {
      sum += slice[index_pixel];
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    slice_sums[index_slice] = sum_cache[0];
  }
}

__global__ void kernel_calculate_responsabilities_sparse(const int *const pattern_start_indices,
							 const int *const pattern_indices,
							 const float *const pattern_values,
							 const float *const slices,
							 const int number_of_pixels,
							 float *const responsabilities,
							 const float *const slice_sums,
							 const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int number_of_patterns = gridDim.x;
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const float *const slice = &slices[number_of_pixels*index_slice];
  
  int index_pixel;
  float sum = 0.;
  for (int index = pattern_start_indices[index_pattern]+threadIdx.x;
       index < pattern_start_indices[index_pattern+1];
       index += blockDim.x) {
    index_pixel = pattern_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += (((int) pattern_values[index]) *
	      logf(slice[index_pixel]) -
	      log_factorial_table[(int)pattern_values[index]]);
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice] + sum_cache[0];
  }
}

/* Need to calculate slice sums before calling this kernel. But it can be done in python */
void calculate_responsabilities_sparse(const int *const pattern_start_indices,
				       const int *const pattern_indices,
				       const float *const pattern_values,
				       const int number_of_patterns,
				       const float *const slices,
				       const int number_of_rotations,
				       const int image_x,
				       const int image_y,
				       float *const responsabilities,
				       float *const slice_sums,
				       const float *const log_factorial_table)
{
  const int nblocks_sum_slices = number_of_rotations;
  const int nthreads_sum_slices = NTHREADS;
  kernel_sum_slices<<<nblocks_sum_slices, nthreads_sum_slices>>>(slices,
								 image_x*image_y,
								 slice_sums);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
  
  const dim3 nblocks_calc_resp(number_of_patterns, number_of_rotations);
  const int nthreads_calc_resp = NTHREADS;
  kernel_calculate_responsabilities_sparse<<<nblocks_calc_resp, nthreads_calc_resp>>>(pattern_start_indices,
										      pattern_indices,
										      pattern_values,
										      slices,
										      image_x*image_y,
										      responsabilities,
										      slice_sums,
										      log_factorial_table);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_responsabilities_sparse_scaling(const int *const pattern_start_indices,
								 const int *const pattern_indices,
								 const float *const pattern_values,
								 const float *const slices,
								 const int number_of_pixels,
								 const float *const scaling,
								 float *const responsabilities,
								 const float *const slice_sums,
								 const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int number_of_patterns = gridDim.x;
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const float *const slice = &slices[number_of_pixels*index_slice];
  const float this_scaling = scaling[index_slice*number_of_patterns + index_pattern];

  
  int index_pixel;
  float sum = 0.;
  for (int index = pattern_start_indices[index_pattern]+threadIdx.x; index < pattern_start_indices[index_pattern+1]; index += blockDim.x) {
    index_pixel = pattern_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += ((int) pattern_values[index]) * logf(slice[index_pixel]/this_scaling) - log_factorial_table[(int)pattern_values[index]];
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice]/this_scaling + sum_cache[0];
  }
}

/* Need to calculate slice sums before calling this kernel. But it can be done in python */
void calculate_responsabilities_sparse_scaling(const int *const pattern_start_indices,
					       const int *const pattern_indices,
					       const float *const pattern_values,
					       const int number_of_patterns,
					       const float *const slices,
					       const int number_of_rotations,
					       const int image_x,
					       const int image_y,
					       const float *const scaling,
					       float *const responsabilities,
					       float *const slice_sums,
					       const float *const log_factorial_table)
{
  const int nblocks_sum_slices = number_of_rotations;
  const int nthreads_sum_slices = NTHREADS;
  kernel_sum_slices<<<nblocks_sum_slices, nthreads_sum_slices>>>(slices,
								 image_x*image_y,
								 slice_sums);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
  
  const dim3 nblocks_calc_resp(number_of_patterns, number_of_rotations);
  const int nthreads_calc_resp = NTHREADS;
  kernel_calculate_responsabilities_sparse_scaling<<<nblocks_calc_resp, nthreads_calc_resp>>>(pattern_start_indices,
											      pattern_indices,
											      pattern_values,
											      slices,
											      image_x*image_y,
											      scaling,
											      responsabilities,
											      slice_sums,
											      log_factorial_table);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}
