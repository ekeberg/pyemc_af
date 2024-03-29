#include <Python.h>
#include <emc_cuda.h>	
#include "cuda_tools.h"

using namespace std;

__global__ void kernel_set_to_value(float *const array,
				    const int size,
				    const float value)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < size) {
    array[index] = value;
  }
}

void set_to_value(float *const array,
		  const int size,
		  const float value)
{
  const int nthreads = NTHREADS;
  const int nblocks = (size-1) / nthreads + 1;
  kernel_set_to_value<<<nblocks, nthreads>>>(array, size, value);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_masked_set(float *const array,
				  const int *const mask,
				  const int size,
				  const float value)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < size && mask[index] > 0) {
    array[index] = value;
  }
}

void masked_set(float *const array,
		const int *const mask,
		const int size,
		const float value)
{
  const int nthreads = NTHREADS;
  const int nblocks = (size-1) / nthreads + 1;
  kernel_masked_set<<<nblocks, nthreads>>>(array, mask, size, value);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
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

__device__ void quaternion_multiply(float *const res, const float *const quat1, const float *const quat2)
{
  res[0] = quat1[0]*quat2[0] - quat1[1]*quat2[1] - quat1[2]*quat2[2] - quat1[3]*quat2[3];
  res[1] = quat1[0]*quat2[1] + quat1[1]*quat2[0] + quat1[2]*quat2[3] - quat1[3]*quat2[2];
  res[2] = quat1[0]*quat2[2] - quat1[1]*quat2[3] + quat1[2]*quat2[0] + quat1[3]*quat2[1];
  res[3] = quat1[0]*quat2[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1] + quat1[3]*quat2[0];
}

__device__ void device_interpolate_get_coordinate_weight(const float coordinate,
							 const int side,
							 int *low_coordinate,
							 float *low_weight,
							 float *high_weight,
							 int *out_of_range)
{
  *low_coordinate = (int)ceil(coordinate) - 1;
  *low_weight = ceil(coordinate) - coordinate;
  *high_weight = 1.-*low_weight;
  if (*low_coordinate < -1) {
    *out_of_range = 1;
  } else if (*low_coordinate == -1) {
    *low_weight = 0.;
  } else if (*low_coordinate == side-1) {
    *high_weight = 0.;
  } else if (*low_coordinate > side-1) {
    *out_of_range = 1;
  }
}

__device__ float device_model_get(const float *const model,
				  const int model_x,
				  const int model_y,
				  const int model_z,
				  const float coordinate_x,
				  const float coordinate_y,
				  const float coordinate_z)
{
  int low_x, low_y, low_z;
  float low_weight_x, low_weight_y, low_weight_z;
  float high_weight_x, high_weight_y, high_weight_z;
  int out_of_range = 0;
  device_interpolate_get_coordinate_weight(coordinate_x, model_x,
					   &low_x, &low_weight_x,
					   &high_weight_x, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_y, model_y,
					   &low_y, &low_weight_y,
					   &high_weight_y, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_z, model_z,
					   &low_z, &low_weight_z,
					   &high_weight_z, &out_of_range);

  if (out_of_range != 0) {
    return -1.f;
  } else {
    float interp_sum = 0.;
    float interp_weight = 0.;
    int index_x, index_y, index_z;
    float weight_x, weight_y, weight_z;
    for (index_x = low_x; index_x <= low_x+1; index_x += 1) {
      if (index_x == low_x && low_weight_x == 0.) continue;
      if (index_x == (low_x+1) && high_weight_x == 0.) continue;
      if (index_x == low_x) weight_x = low_weight_x;
      else weight_x = high_weight_x;

      for (index_y = low_y; index_y <= low_y+1; index_y += 1) {
	if (index_y == low_y && low_weight_y == 0.) continue;
	if (index_y == (low_y+1) && high_weight_y == 0.) continue;
	if (index_y == low_y) weight_y = low_weight_y;
	else weight_y = high_weight_y;

	for (index_z = low_z; index_z <= low_z+1; index_z += 1) {
	  if (index_z == low_z && low_weight_z == 0.) continue;
	  if (index_z == (low_z+1) && high_weight_z == 0.) continue;
	  if (index_z == low_z) weight_z = low_weight_z;
	  else weight_z = high_weight_z;

	  if (model[model_z*model_y*index_x + model_z*index_y + index_z] >= 0.) {
	    interp_sum += weight_x*weight_y*weight_z*model[model_z*model_y*index_x + model_z*index_y + index_z];
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

__device__ void device_get_slice(const float *const model,
				 const int model_x,
				 const int model_y,
				 const int model_z,
				 float *const slice,
				 const int image_x,
				 const int image_y,
				 const float *const rotation,
				 const float *const coordinates) {  
  const float *const coordinates_0 = &coordinates[0*image_x*image_y];
  const float *const coordinates_1 = &coordinates[1*image_x*image_y];
  const float *const coordinates_2 = &coordinates[2*image_x*image_y];

  float m00 = (rotation[0]*rotation[0] + rotation[1]*rotation[1] -
	       rotation[2]*rotation[2] - rotation[3]*rotation[3]);
  float m01 = 2.0f*rotation[1]*rotation[2] - 2.0f*rotation[0]*rotation[3];
  float m02 = 2.0f*rotation[1]*rotation[3] + 2.0f*rotation[0]*rotation[2];
  float m10 = 2.0f*rotation[1]*rotation[2] + 2.0f*rotation[0]*rotation[3];
  float m11 = (rotation[0]*rotation[0] - rotation[1]*rotation[1] +
	       rotation[2]*rotation[2] - rotation[3]*rotation[3]);
  float m12 = 2.0f*rotation[2]*rotation[3] - 2.0f*rotation[0]*rotation[1];
  float m20 = 2.0f*rotation[1]*rotation[3] - 2.0f*rotation[0]*rotation[2];
  float m21 = 2.0f*rotation[2]*rotation[3] + 2.0f*rotation[0]*rotation[1];
  float m22 = (rotation[0]*rotation[0] - rotation[1]*rotation[1] -
	       rotation[2]*rotation[2] + rotation[3]*rotation[3]);

  float new_x, new_y, new_z;
  for (int x = 0; x < image_x; x++) {
    for (int y = threadIdx.x; y < image_y; y+=blockDim.x) {
      /* This is just a matrix multiplication with rotation */
      new_x = (m00*coordinates_0[x*image_y+y] +
	       m01*coordinates_1[x*image_y+y] +
	       m02*coordinates_2[x*image_y+y] +
	       model_x/2.0 - 0.5);
      new_y = (m10*coordinates_0[x*image_y+y] +
	       m11*coordinates_1[x*image_y+y] +
	       m12*coordinates_2[x*image_y+y] +
	       model_y/2.0 - 0.5);
      new_z = (m20*coordinates_0[x*image_y+y] +
	       m21*coordinates_1[x*image_y+y] +
	       m22*coordinates_2[x*image_y+y] +
	       model_z/2.0 - 0.5);

      slice[x*image_y+y] = device_model_get(model, model_x, model_y, model_z, new_x, new_y, new_z);
    }
  }
}

__global__ void kernel_expand_model(const float *const model,
				    const int model_x,
				    const int model_y,
				    const int model_z,
				    float *const slices,
				    const int image_x,
				    const int image_y,
				    const float *const rotations,
				    const float *const coordinates)
{
  const int rotation_index = blockIdx.x;
  device_get_slice(model,
		   model_x,
		   model_y,
		   model_z,
		   &slices[image_x*image_y*rotation_index],
		   image_x,
		   image_y,
		   &rotations[4*rotation_index],
		   coordinates);
}
				    

void expand_model(const float *const model,
		  const int model_x,
		  const int model_y,
		  const int model_z,
		  float *const slices,
		  const int image_x,
		  const int image_y,
		  const float *const rotations,
		  const int number_of_rotations,
		  const float *const coordinates)
{
  int nblocks = number_of_rotations;
  int nthreads = NTHREADS;
  kernel_expand_model<<<nblocks, nthreads>>>(model,
					     model_x,
					     model_y,
					     model_z,
					     slices,
					     image_x,
					     image_y,
					     rotations,
					     coordinates);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__device__ void device_model_set(float *const model,
				 float *const model_weights,
				 const int model_x,
				 const int model_y,
				 const int model_z,
				 const float coordinate_x,
				 const float coordinate_y,
				 const float coordinate_z,
				 const float value,
				 const float value_weight)
{
  int low_x, low_y, low_z;
  float low_weight_x, low_weight_y, low_weight_z;
  float high_weight_x, high_weight_y, high_weight_z;
  int out_of_range = 0;
  device_interpolate_get_coordinate_weight(coordinate_x, model_x,
					   &low_x, &low_weight_x,
					   &high_weight_x, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_y, model_y,
					   &low_y, &low_weight_y,
					   &high_weight_y, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_z, model_z,
					   &low_z, &low_weight_z,
					   &high_weight_z, &out_of_range);

  if (out_of_range == 0) {
    int index_x, index_y, index_z;
    float weight_x, weight_y, weight_z;
    for (index_x = low_x; index_x <= low_x+1; index_x += 1) {
      if (index_x == low_x && low_weight_x == 0.) continue;
      if (index_x == (low_x+1) && high_weight_x == 0.) continue;
      if (index_x == low_x) weight_x = low_weight_x;
      else weight_x = high_weight_x;

      for (index_y = low_y; index_y <= low_y+1; index_y += 1) {
	if (index_y == low_y && low_weight_y == 0.) continue;
	if (index_y == (low_y+1) && high_weight_y == 0.) continue;
	if (index_y == low_y) weight_y = low_weight_y;
	else weight_y = high_weight_y;

	for (index_z = low_z; index_z <= low_z+1; index_z += 1) {
	  if (index_z == low_z && low_weight_z == 0.) continue;
	  if (index_z == (low_z+1) && high_weight_z == 0.) continue;
	  if (index_z == low_z) weight_z = low_weight_z;
	  else weight_z = high_weight_z;
	  
	  atomicAdd(&model[model_z*model_y*index_x + model_z*index_y + index_z],
		    weight_x*weight_y*weight_z*value_weight*value);
	  atomicAdd(&model_weights[model_z*model_y*index_x + model_z*index_y + index_z],
		    weight_x*weight_y*weight_z*value_weight);
	}
      }
    }
  }
}

__device__ void device_model_set_nn(float *const model,
				    float *const model_weights,
				    const int model_x,
				    const int model_y,
				    const int model_z,
				    const float coordinate_x,
				    const float coordinate_y,
				    const float coordinate_z,
				    const float value,
				    const float value_weight)
{
  int index_x = (int) (coordinate_x + 0.5);
  int index_y = (int) (coordinate_y + 0.5);
  int index_z = (int) (coordinate_z + 0.5);
  if (index_x >= 0 && index_x < model_x &&
      index_y >= 0 && index_y < model_y &&
      index_z >= 0 && index_z < model_z) {
    atomicAdd(&model[model_z*model_y*index_x + model_z*index_y + index_z],
	      value_weight*value);
    atomicAdd(&model_weights[model_z*model_y*index_x + model_z*index_y + index_z],
	      value_weight);
  }
}

__device__ void device_insert_slice(float *const model,
				    float *const model_weights,
				    const int model_x,
				    const int model_y,
				    const int model_z,
				    const float *const slice,
				    const int image_x,
				    const int image_y,
				    const float slice_weight,
				    const float *const rotation,
				    const float *const coordinates,
				    const int interpolation)
{
  const float *const coordinates_0 = &coordinates[0*image_x*image_y];
  const float *const coordinates_1 = &coordinates[1*image_x*image_y];
  const float *const coordinates_2 = &coordinates[2*image_x*image_y];

  float m00 = (rotation[0]*rotation[0] + rotation[1]*rotation[1] -
	       rotation[2]*rotation[2] - rotation[3]*rotation[3]);
  float m01 = 2.0f*rotation[1]*rotation[2] - 2.0f*rotation[0]*rotation[3];
  float m02 = 2.0f*rotation[1]*rotation[3] + 2.0f*rotation[0]*rotation[2];
  float m10 = 2.0f*rotation[1]*rotation[2] + 2.0f*rotation[0]*rotation[3];
  float m11 = (rotation[0]*rotation[0] - rotation[1]*rotation[1] +
	       rotation[2]*rotation[2] - rotation[3]*rotation[3]);
  float m12 = 2.0f*rotation[2]*rotation[3] - 2.0f*rotation[0]*rotation[1];
  float m20 = 2.0f*rotation[1]*rotation[3] - 2.0f*rotation[0]*rotation[2];
  float m21 = 2.0f*rotation[2]*rotation[3] + 2.0f*rotation[0]*rotation[1];
  float m22 = (rotation[0]*rotation[0] - rotation[1]*rotation[1] -
	       rotation[2]*rotation[2] + rotation[3]*rotation[3]);

  float new_x, new_y, new_z;
  for (int x = 0; x < image_x; x++) {
    for (int y = threadIdx.x; y < image_y; y+=blockDim.x) {
      if (slice[x*image_y+y] >= 0.) {
	/* This is just a matrix multiplication with rotation */
	new_x = (m00*coordinates_0[x*image_y+y] +
		 m01*coordinates_1[x*image_y+y] +
		 m02*coordinates_2[x*image_y+y] +
		 model_x/2.0 - 0.5);
	new_y = (m10*coordinates_0[x*image_y+y] +
		 m11*coordinates_1[x*image_y+y] +
		 m12*coordinates_2[x*image_y+y] +
		 model_y/2.0 - 0.5);
	new_z = (m20*coordinates_0[x*image_y+y] +
		 m21*coordinates_1[x*image_y+y] +
		 m22*coordinates_2[x*image_y+y] +
		 model_z/2.0 - 0.5);

	if (interpolation == 0) {
	  device_model_set_nn(model, model_weights,
			      model_x, model_y, model_z,
			      new_x, new_y, new_z,
			      slice[x*image_y+y], slice_weight);
	} else {
	  device_model_set(model, model_weights,
			   model_x, model_y, model_z,
			   new_x, new_y, new_z,
			   slice[x*image_y+y], slice_weight);
	}
      }
    }
  }
}


__global__ void kernel_insert_slices(float *const model,
				     float *const model_weights,
				     const int model_x,
				     const int model_y,
				     const int model_z,
				     const float *const slices,
				     const int image_x,
				     const int image_y,
				     const float *const slice_weights,
				     const float *const rotations,
				     const float *const coordinates,
				     const int interpolation) {
  const int rotation_index = blockIdx.x;
  device_insert_slice(model,
		      model_weights,
		      model_x,
		      model_y,
		      model_z,
		      &slices[image_x*image_y*rotation_index],
		      image_x,
		      image_y,
		      slice_weights[rotation_index],
		      &rotations[4*rotation_index],
		      coordinates,
		      interpolation);
}

void insert_slices(float *const model,
		   float *const model_weights,
		   const int model_x,
		   const int model_y,
		   const int model_z,
		   const float *const slices,
		   const int image_x,
		   const int image_y,
		   const float *const slice_weights,
		   const float *const rotations,
		   const int number_of_rotations,
		   const float *const coordinates,
		   const int interpolation)
{
  int nblocks = number_of_rotations;
  int nthreads = NTHREADS;
  kernel_insert_slices<<<nblocks, nthreads>>>(model,
					      model_weights,
					      model_x,
					      model_y,
					      model_z,
					      slices,
					      image_x,
					      image_y,
					      slice_weights,
					      rotations,
					      coordinates,
					      interpolation);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__device__ void device_insert_slice_partial(float *const model,
					    float *const model_weights,
					    const int model_x_tot,
					    const int model_x_min,
					    const int model_x_max,
					    const int model_y_tot,
					    const int model_y_min,
					    const int model_y_max,
					    const int model_z_tot,
					    const int model_z_min,
					    const int model_z_max,
					    const float *const slice,
					    const int image_x,
					    const int image_y,
					    const float slice_weight,
					    const float *const rotation,
					    const float *const coordinates,
					    const int interpolation)
{
  const float *const coordinates_0 = &coordinates[0*image_x*image_y];
  const float *const coordinates_1 = &coordinates[1*image_x*image_y];
  const float *const coordinates_2 = &coordinates[2*image_x*image_y];

  float m00 = (rotation[0]*rotation[0] + rotation[1]*rotation[1] -
	       rotation[2]*rotation[2] - rotation[3]*rotation[3]);
  float m01 = 2.0f*rotation[1]*rotation[2] - 2.0f*rotation[0]*rotation[3];
  float m02 = 2.0f*rotation[1]*rotation[3] + 2.0f*rotation[0]*rotation[2];
  float m10 = 2.0f*rotation[1]*rotation[2] + 2.0f*rotation[0]*rotation[3];
  float m11 = (rotation[0]*rotation[0] - rotation[1]*rotation[1] +
	       rotation[2]*rotation[2] - rotation[3]*rotation[3]);
  float m12 = 2.0f*rotation[2]*rotation[3] - 2.0f*rotation[0]*rotation[1];
  float m20 = 2.0f*rotation[1]*rotation[3] - 2.0f*rotation[0]*rotation[2];
  float m21 = 2.0f*rotation[2]*rotation[3] + 2.0f*rotation[0]*rotation[1];
  float m22 = (rotation[0]*rotation[0] - rotation[1]*rotation[1] -
	       rotation[2]*rotation[2] + rotation[3]*rotation[3]);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("inside kernel\n");
    /*
      printf("%i %i %i\n", model_x_tot, model_x_min, model_x_max);
      printf("%i %i %i\n", model_y_tot, model_y_min, model_y_max);
      printf("%i %i %i\n", model_z_tot, model_z_min, model_z_max);
    */
    printf("%i %i %i\n", model_x_max-model_x_min, model_y_max-model_y_min, model_z_max-model_z_min);
  }
  
  float new_x, new_y, new_z;
  for (int x = 0; x < image_x; x++) {
    for (int y = threadIdx.x; y < image_y; y+=blockDim.x) {
      if (slice[x*image_y+y] >= 0.) {
	/* This is just a matrix multiplication with rotation */
	new_x = (m00*coordinates_0[x*image_y+y] +
		 m01*coordinates_1[x*image_y+y] +
		 m02*coordinates_2[x*image_y+y] +
		 model_x_tot/2.0 - 0.5);
	new_y = (m10*coordinates_0[x*image_y+y] +
		 m11*coordinates_1[x*image_y+y] +
		 m12*coordinates_2[x*image_y+y] +
		 model_y_tot/2.0 - 0.5);
	new_z = (m20*coordinates_0[x*image_y+y] +
		 m21*coordinates_1[x*image_y+y] +
		 m22*coordinates_2[x*image_y+y] +
		 model_z_tot/2.0 - 0.5);

	if (blockIdx.x == 0 && x == 10 && y == 10) {
	  printf("%g %g %g\n", new_x, new_y, new_z);
	  printf("%g %g %g\n", new_x-(float)model_x_min, new_y-(float)model_y_min, new_z-(float)model_z_min);
	}

	if (interpolation == 0) {
	  device_model_set_nn(model,
			      model_weights,
			      model_x_max-model_x_min,
			      model_y_max-model_y_min,
			      model_z_max-model_z_min,
			      new_x-(float)model_x_min,
			      new_y-(float)model_y_min,
			      new_z-(float)model_z_min,
			      slice[x*image_y+y],
			      slice_weight);
	} else {
	  device_model_set(model,
			   model_weights,
			   model_x_max-model_x_min,
			   model_y_max-model_y_min,
			   model_z_max-model_z_min,
			   new_x-(float)model_x_min,
			   new_y-(float)model_y_min,
			   new_z-(float)model_z_min,
			   slice[x*image_y+y],
			   slice_weight);
	}
      }
    }
  }
}

__global__ void kernel_insert_slices_partial(float *const model,
					     float *const model_weights,
					     const int model_x_tot,
					     const int model_x_min,
					     const int model_x_max,
					     const int model_y_tot,
					     const int model_y_min,
					     const int model_y_max,
					     const int model_z_tot,
					     const int model_z_min,
					     const int model_z_max,
					     const float *const slices,
					     const int image_x,
					     const int image_y,
					     const float *const slice_weights,
					     const float *const rotations,
					     const float *const coordinates,
					     const int interpolation) {
  const int rotation_index = blockIdx.x;
  device_insert_slice_partial(model,
			      model_weights,
			      model_x_tot,
			      model_x_min,
			      model_x_max,
			      model_y_tot,
			      model_y_min,
			      model_y_max,
			      model_z_tot,
			      model_z_min,
			      model_z_max,
			      &slices[image_x*image_y*rotation_index],
			      image_x,
			      image_y,
			      slice_weights[rotation_index],
			      &rotations[4*rotation_index],
			      coordinates,
			      interpolation);
}

void insert_slices_partial(float *const model,
			   float *const model_weights,
			   const int model_x_tot,
			   const int model_x_min,
			   const int model_x_max,
			   const int model_y_tot,
			   const int model_y_min,
			   const int model_y_max,
			   const int model_z_tot,
			   const int model_z_min,
			   const int model_z_max,
			   const float *const slices,
			   const int image_x,
			   const int image_y,
			   const float *const slice_weights,
			   const float *const rotations,
			   const int number_of_rotations,
			   const float *const coordinates,
			   const int interpolation)
{
  int nblocks = number_of_rotations;
  int nthreads = NTHREADS;
  kernel_insert_slices_partial<<<nblocks, nthreads>>>(model,
						      model_weights,
						      model_x_tot,
						      model_x_min,
						      model_x_max,
						      model_y_tot,
						      model_y_min,
						      model_y_max,
						      model_z_tot,
						      model_z_min,
						      model_z_max,
						      slices,
						      image_x,
						      image_y,
						      slice_weights,
						      rotations,
						      coordinates,
						      interpolation);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}


__device__ float device_compare_balanced(const float expected_value,
					 const float measured_value) 
{
  return pow(expected_value - measured_value, 2) / expected_value;
}

__global__ void kernel_rotate_model(const float *const model,
				    float *const rotated_model,
				    const int model_x,
				    const int model_y,
				    const int model_z,
				    const float *const rotation) {

  __shared__ float rotation_matrix[9];
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (threadIdx.x == 0) {
    rotation_matrix[0] = rotation[0]*rotation[0] + rotation[1]*rotation[1] - rotation[2]*rotation[2] - rotation[3]*rotation[3]; // 00
    rotation_matrix[1] = 2.0f*rotation[1]*rotation[2] - 2.0f*rotation[0]*rotation[3]; // 01
    rotation_matrix[2] = 2.0f*rotation[1]*rotation[3] + 2.0f*rotation[0]*rotation[2]; // 02
    rotation_matrix[3] = 2.0f*rotation[1]*rotation[2] + 2.0f*rotation[0]*rotation[3]; // 10
    rotation_matrix[4] = rotation[0]*rotation[0] - rotation[1]*rotation[1] + rotation[2]*rotation[2] - rotation[3]*rotation[3]; // 11
    rotation_matrix[5] = 2.0f*rotation[2]*rotation[3] - 2.0f*rotation[0]*rotation[1]; // 12
    rotation_matrix[6] = 2.0f*rotation[1]*rotation[3] - 2.0f*rotation[0]*rotation[2]; // 20
    rotation_matrix[7] = 2.0f*rotation[2]*rotation[3] + 2.0f*rotation[0]*rotation[1]; // 21
    rotation_matrix[8] = rotation[0]*rotation[0] - rotation[1]*rotation[1] - rotation[2]*rotation[2] + rotation[3]*rotation[3]; // 22
  }
  __syncthreads();

  if (index < model_x*model_y*model_z) {

    float start_x = ((float) ((index % (model_x*model_y)) % model_x)) - model_x/2. + 0.5;
    float start_y = ((float) ((index / model_x) % model_y)) - model_y/2. + 0.5;
    float start_z = ((float) (index / (model_x*model_y))) - model_z/2. + 0.5;

    /*
      float start_x = ((float) ((index % (model_x*model_y)) % model_x));
      float start_y = ((float) ((index / model_x) % model_y));
      float start_z = ((float) (index / (model_x*model_y)));
    */
    float new_x, new_y, new_z;
    /* This is just a matrix multiplication with rotation */
    new_x = model_x/2. - 0.5 + (rotation_matrix[0]*start_x +
				rotation_matrix[1]*start_y +
				rotation_matrix[2]*start_z);
    new_y = model_y/2. - 0.5 + (rotation_matrix[3]*start_x +
				rotation_matrix[4]*start_y +
				rotation_matrix[5]*start_z);
    new_z = model_z/2. - 0.5 + (rotation_matrix[6]*start_x +
				rotation_matrix[7]*start_y +
				rotation_matrix[8]*start_z);
    rotated_model[index] = device_model_get(model,
					    model_x, model_y, model_z,
					    new_x, new_y, new_z);
  }
}

void rotate_model(const float *const model,
		  float *const rotated_model,
		  const int model_x,
		  const int model_y,
		  const int model_z,
		  const float *const rotation)
{
  const int nthreads = NTHREADS;
  const int nblocks = (model_x*model_y*model_z-1) / nthreads + 1;
  kernel_rotate_model<<<nblocks, nthreads>>>(model,
					     rotated_model,
					     model_x,
					     model_y,
					     model_z,
					     rotation);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}
