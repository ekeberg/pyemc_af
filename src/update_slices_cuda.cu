#include <Python.h>
#include <emc_cuda.h>	
#include "cuda_tools.h"

using namespace std;

__global__ void kernel_normalize_slices(float *const slices,
					const float *responsabilities,
					const int number_of_pixels,
					const int number_of_patterns)
{
  __shared__ float normalization_factor_cache[NTHREADS];
  const int index_rotation = blockIdx.x;

  normalization_factor_cache[threadIdx.x] = 0.;
  for (int index_pattern = threadIdx.x;
       index_pattern < number_of_patterns;
       index_pattern += blockDim.x) {
    float this_resp = responsabilities[index_rotation*number_of_patterns + index_pattern];
    normalization_factor_cache[threadIdx.x] += this_resp;
  }
  inblock_reduce(normalization_factor_cache);
  float normalization_factor;
  if (normalization_factor_cache[0] > -10.) {
    normalization_factor = 1./normalization_factor_cache[0];
    //normalization_factor = normalization_factor_cache[0];
  } else {
    normalization_factor = 0.;
    //normalization_factor = normalization_factor_cache[0];
  }
  for (int index_pixel = threadIdx.x; index_pixel < number_of_pixels; index_pixel += blockDim.x) {
    slices[index_rotation*number_of_pixels + index_pixel] *= normalization_factor;
    //slices[index_rotation*number_of_pixels + index_pixel] *= 1./normalization_factor;
  }
}

__global__ void kernel_update_slices(float *const slices,
				     const float *const patterns,
				     const int number_of_patterns,
				     const int number_of_pixels,
				     const float *const responsabilities)
{
  const int index_rotation = blockIdx.x;
  float sum;
  float weight;
  for (int pixel_index = threadIdx.x;
       pixel_index < number_of_pixels;
       pixel_index += blockDim.x) {
    sum = 0.;
    weight = 0.;
    for (int pattern_index = 0; 
	 pattern_index < number_of_patterns; 
	 pattern_index++) {
      if (patterns[pattern_index*number_of_pixels + pixel_index] >= 0.) {
	sum += (patterns[pattern_index*number_of_pixels + pixel_index] *
		responsabilities[index_rotation*number_of_patterns + pattern_index]);
	weight += responsabilities[index_rotation*number_of_patterns + pattern_index];
      }
    }
    if (weight > 0.) {
      slices[index_rotation*number_of_pixels + pixel_index] = sum / weight;
    } else {
      slices[index_rotation*number_of_pixels + pixel_index] = -1.;
    }
  }

  /* const int index_rotation = blockIdx.x; */
  /* float sum; */
  /* float weight; */
  /* for (int pixel_index = 0; pixel_index < number_of_pixels; pixel_index++) { */
  /*   sum = 0.; */
  /*   weight = 0.; */
  /*   for (int pattern_index = threadIdx.x;  */
  /* 	 pattern_index < number_of_patterns;  */
  /* 	 pattern_index += blockDim.x) { */
  /*     if (patterns[pattern_index*number_of_pixels + pixel_index] >= 0.) { */
  /* 	sum += (patterns[pattern_index*number_of_pixels + pixel_index] * */
  /* 		responsabilities[index_rotation*number_of_patterns + pattern_index]); */
  /* 	weight += responsabilities[index_rotation*number_of_patterns + pattern_index]; */
  /*     } */
  /*   } */
  /*   if (weight > 0.) { */
  /*     slices[index_rotation*number_of_pixels + pixel_index] = sum / weight; */
  /*   } else { */
  /*     slices[index_rotation*number_of_pixels + pixel_index] = -1.; */
  /*   } */
  /* } */
}

void update_slices(float *const slices,
		   const int number_of_rotations,
		   const float *const patterns,
		   const int number_of_patterns,
		   const int image_x,
		   const int image_y,
		   const float *const responsabilities)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices<<<nblocks, nthreads>>>(slices,
					      patterns,
					      number_of_patterns,
					      image_x*image_y,
					      responsabilities);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_update_slices_scaling(float *const slices,
					     const float *const patterns,
					     const int number_of_patterns,
					     const int number_of_pixels,
					     const float *const responsabilities,
					     const float *const scaling)
{
  const int index_rotation = blockIdx.x;
  float sum;
  float weight;
  for (int pixel_index = threadIdx.x; pixel_index < number_of_pixels; pixel_index += blockDim.x) {
    sum = 0.;
    weight = 0.;
    for (int pattern_index = 0;
	 pattern_index < number_of_patterns;
	 pattern_index += 1) {
      if (patterns[pattern_index*number_of_pixels + pixel_index] >= 0.) {
	sum += (patterns[pattern_index*number_of_pixels + pixel_index] *
		scaling[index_rotation*number_of_patterns + pattern_index] *
		responsabilities[index_rotation*number_of_patterns + pattern_index]);
	weight += responsabilities[index_rotation*number_of_patterns + pattern_index];
      }
    }
    if (weight > 0.) {
      slices[index_rotation*number_of_pixels + pixel_index] = sum / weight;
    } else {
      slices[index_rotation*number_of_pixels + pixel_index] = -1.;
    }
  }
}

void update_slices_scaling(float *const slices,
			   const int number_of_rotations,
			   const float *const patterns,
			   const int number_of_patterns,
			   const int image_x,
			   const int image_y,
			   const float *const responsabilities,
			   const float *const scaling)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices_scaling<<<nblocks, nthreads>>>(slices,
						      patterns,
						      number_of_patterns,
						      image_x*image_y,
						      responsabilities,
						      scaling);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}


__global__ void kernel_update_slices_per_pattern_scaling(float *const slices,
							 const float *const patterns,
							 const int number_of_patterns,
							 const int number_of_pixels,
							 const float *const responsabilities,
							 const float *const scaling)
{
  const int index_rotation = blockIdx.x;
  float sum;
  float weight;
  for (int pixel_index = threadIdx.x; pixel_index < number_of_pixels; pixel_index += blockDim.x) {
    sum = 0.;
    weight = 0.;
    for (int pattern_index = 0; pattern_index < number_of_patterns; pattern_index++) {
      if (patterns[pattern_index*number_of_pixels + pixel_index] >= 0.) {
	sum += (patterns[pattern_index*number_of_pixels + pixel_index] *
		scaling[pattern_index] *
		responsabilities[index_rotation*number_of_patterns + pattern_index]);
	weight += responsabilities[index_rotation*number_of_patterns + pattern_index];
      }
    }
    if (weight > 0.) {
      slices[index_rotation*number_of_pixels + pixel_index] = sum / weight;
    } else {
      slices[index_rotation*number_of_pixels + pixel_index] = -1.;
    }
  }
}

void update_slices_per_pattern_scaling(float *const slices,
				       const int number_of_rotations,
				       const float *const patterns,
				       const int number_of_patterns,
				       const int image_x,
				       const int image_y,
				       const float *const responsabilities,
				       const float *const scaling)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices_per_pattern_scaling<<<nblocks, nthreads>>>(slices,
								  patterns,
								  number_of_patterns,
								  image_x*image_y,
								  responsabilities,
								  scaling);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

/* This can't handle masks att the moment. Need to think about how to handle masked out data in the sparse implemepntation
 */
__global__ void kernel_update_slices_sparse(float *const slices,
					    const int number_of_pixels,
					    const int *const pattern_start_indices,
					    const int *const pattern_indices,
					    const float *const pattern_values,
					    const int number_of_patterns,
					    const float *const responsabilities,
					    const float resp_threshold)
{
  //const int number_of_rotations = gridDim.x;
  const int index_rotation = blockIdx.x;
  
  int index_pixel;
  
  for (int index_pixel = threadIdx.x; index_pixel < number_of_pixels; index_pixel += blockDim.x) {
    slices[index_rotation*number_of_pixels + index_pixel] = 0.0;
  }
  __syncthreads();
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    float this_resp = responsabilities[index_rotation*number_of_patterns + index_pattern];
    for (int value_index = pattern_start_indices[index_pattern] + threadIdx.x;
    	 value_index < pattern_start_indices[index_pattern+1];
    	 value_index += blockDim.x) {
      index_pixel = pattern_indices[value_index];
      if (this_resp > resp_threshold) {
	atomicAdd(&slices[index_rotation*number_of_pixels + index_pixel],
		  pattern_values[value_index] * this_resp);
      }
    }
  }
}

void update_slices_sparse(float *const slices,
			  const int number_of_rotations,
			  const int *const pattern_start_indices,
			  const int *const pattern_indices,
			  const float *const pattern_values,
			  const int number_of_patterns,
			  const int image_x,
			  const int image_y,
			  const float *const responsabilities,
			  const float resp_threshold){
  /* printf("Update slices sparse (%d, %d)!\n", number_of_rotations, NTHREADS); */
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices_sparse<<<nblocks, nthreads>>>(slices,
						     image_x*image_y,
						     pattern_start_indices,
						     pattern_indices,
						     pattern_values,
						     number_of_patterns,
						     responsabilities,
						     resp_threshold);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());

  const int nblocks_normalize = number_of_rotations;
  kernel_normalize_slices<<<nblocks_normalize, nthreads>>>(slices, responsabilities, image_x*image_y, number_of_patterns);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_update_slices_sparse_scaling(float *const slices,
						    const int number_of_pixels,
						    const int *const pattern_start_indices,
						    const int *const pattern_indices,
						    const float *const pattern_values,
						    const int number_of_patterns,
						    const float *const responsabilities,
						    const float resp_threshold,
						    const float *const scaling)
{
  //const int number_of_rotations = gridDim.x;
  const int index_rotation = blockIdx.x;
  //const int index_pattern = blockIdx.x;

  int index_pixel;

  for (int index_pixel = threadIdx.x; index_pixel < number_of_pixels; index_pixel += blockDim.x) {
    slices[index_rotation*number_of_pixels + index_pixel] = 0.;
  }
  __syncthreads();
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern += 1) {
  /* int index_pattern = blockIdx.x; */
    float this_resp = responsabilities[index_rotation*number_of_patterns + index_pattern];
    float this_scaling = scaling[index_rotation*number_of_patterns + index_pattern];
    if (this_resp > resp_threshold) {
      for (int value_index = pattern_start_indices[index_pattern]+threadIdx.x;
	   value_index < pattern_start_indices[index_pattern+1];
	   value_index += blockDim.x) {
	index_pixel = pattern_indices[value_index];
	atomicAdd(&slices[index_rotation*number_of_pixels + index_pixel],
		  pattern_values[value_index] * this_scaling * this_resp);
      }
    }
  }
  /* float pixel_sum; */
  /* for (int pixel_index = threadIdx.x; pixel_index < number_of_pixels; pixel_index += blockDim.x) { */
  /*   for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern += 1) { */
  /*     float this_resp = responsabilities[index_rotation*number_of_patterns + index_pattern]; */
  /*     float this_scaling = scaling[index_rotation*number_of_patterns + index_pattern]; */
  /*     pixel_sum = 0.; */
  /*     for (int sparse_index = pattern_start_indices[index_pattern]; */
  /* 	   sparse_index < pattern_start_indices[index_pattern+1]; */
  /* 	   sparse_index += 1) { */
  /* 	if (pattern_indices[sparse_index] == pixel_index) { */
  /* 	  pixel_sum += pattern_values[sparse_index]*this_scaling*this_resp; */
  /* 	} */
  /*     } */
  /*   } */
  /*   slices[index_rotation*number_of_pixels + index_pixel] = pixel_sum; */
  /* } */
}

void update_slices_sparse_scaling(float *const slices,
				  const int number_of_rotations,
				  const int *const pattern_start_indices,
				  const int *const pattern_indices,
				  const float *const pattern_values,
				  const int number_of_patterns,
				  const int image_x,
				  const int image_y,
				  const float *const responsabilities,
				  const float resp_threshold,
				  const float *const scaling)
{
  const int nblocks = number_of_rotations;
  /* dim3 nblocks(number_of_patterns, number_of_rotations); */
  const int nthreads = NTHREADS;
  kernel_update_slices_sparse_scaling<<<nblocks, nthreads>>>(slices,
  							     image_x*image_y,
  							     pattern_start_indices,
  							     pattern_indices,
  							     pattern_values,
  							     number_of_patterns,
  							     responsabilities,
							     resp_threshold,
  							     scaling);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());

  const int nblocks_normalize = number_of_rotations;
  kernel_normalize_slices<<<nblocks_normalize, nthreads>>>(slices, responsabilities, image_x*image_y, number_of_patterns);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
  
}

__global__ void kernel_update_slices_sparse_per_pattern_scaling(float *const slices,
								const int number_of_pixels,
								const int *const pattern_start_indices,
								const int *const pattern_indices,
								const float *const pattern_values,
								const int number_of_patterns,
								const float *const responsabilities,
								const float *const scaling)
{
  __shared__ float normalization_factor_cache[NTHREADS];
  //const int number_of_rotations = gridDim.x;
  const int index_rotation = blockIdx.x;

  int index_pixel;

  for (int index_pixel = threadIdx.x; index_pixel < number_of_pixels; index_pixel += blockDim.x) {
    slices[index_rotation*number_of_pixels + index_pixel] = 0.;
  }
  __syncthreads();

  /* for (int index_pattern = threadIdx.x; index_pattern < number_of_patterns; index_pattern += blockDim.x) { */
  /*   for (int value_index = pattern_start_indices[index_pattern]; value_index < pattern_start_indices[index_pattern+1]; value_index += 1) { */
  /*     index_pixel = pattern_indices[value_index]; */
  /*     atomicAdd(&slices[index_rotation*number_of_pixels + index_pixel], */
  /* 		pattern_values[value_index] * scaling[index_pattern] * */
  /* 		responsabilities[index_rotation*number_of_patterns + index_pattern]); */
  /*   } */
  /* } */

  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int value_index = pattern_start_indices[index_pattern] + threadIdx.x;
	 value_index < pattern_start_indices[index_pattern+1];
	 value_index += blockDim.x) {
      index_pixel = pattern_indices[value_index];
      slices[index_rotation*number_of_pixels + index_pixel] += (pattern_values[value_index] *
								scaling[index_pattern] *
								responsabilities[index_rotation*number_of_patterns + index_pattern]);
    }
  }

  
  normalization_factor_cache[threadIdx.x] = 0.;
  for (int index_pattern = threadIdx.x; index_pattern < number_of_patterns; index_pattern += blockDim.x) {
    normalization_factor_cache[threadIdx.x] += responsabilities[index_rotation*number_of_patterns + index_pattern];
  }
  inblock_reduce(normalization_factor_cache);
  float normalization_factor = normalization_factor_cache[0];
  for (int index_pixel = threadIdx.x; index_pixel < number_of_pixels; index_pixel += blockDim.x) {
    slices[index_rotation*number_of_pixels + index_pixel] *= 1./normalization_factor;
  }
}

void update_slices_sparse_per_pattern_scaling(float *const slices,
					      const int number_of_rotations,
					      const int *const pattern_start_indices,
					      const int *const pattern_indices,
					      const float *const pattern_values,
					      const int number_of_patterns,
					      const int image_x,
					      const int image_y,
					      const float *const responsabilities,
					      const float *const scaling)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices_sparse_per_pattern_scaling<<<nblocks, nthreads>>>(slices,
									 image_x*image_y,
									 pattern_start_indices,
									 pattern_indices,
									 pattern_values,
									 number_of_patterns,
									 responsabilities,
									 scaling);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}
