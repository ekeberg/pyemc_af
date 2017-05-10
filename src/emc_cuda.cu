#include <Python.h>
#include <emc_cuda.h>	

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

__global__ void kernel_set_to_value(float *const array, const int size, const float value)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < size) {
    array[index] = value;
  }
}

void set_to_value(float *const array, const int size, const float value)
{
  const int nthreads = NTHREADS;
  const int nblocks = (size-1) / nthreads + 1;
  kernel_set_to_value<<<nblocks, nthreads>>>(array, size, value);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_masked_set(float *const array, const int *const mask, const int size, const float value)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < size && mask[index] > 0) {
    array[index] = value;
  }
}

void masked_set(float *const array, const int *const mask, const int size, const float value)
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

/*
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
*/
/* We should probably change how edges are handled in the interpolation to make it consistent when working with tiles. */
__device__ void device_interpolate_get_coordinate_weight(const float coordinate, const int side,
							 int *low_coordinate, float *low_weight, float *high_weight, int *out_of_range)
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

__device__ float device_model_get(const float *const model, const int model_x, const int model_y, const int model_z,
				  const float coordinate_x, const float coordinate_y, const float coordinate_z)
{
  int low_x, low_y, low_z;
  float low_weight_x, low_weight_y, low_weight_z;
  float high_weight_x, high_weight_y, high_weight_z;
  int out_of_range = 0;
  device_interpolate_get_coordinate_weight(coordinate_x, model_x, &low_x, &low_weight_x, &high_weight_x, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_y, model_y, &low_y, &low_weight_y, &high_weight_y, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_z, model_z, &low_z, &low_weight_z, &high_weight_z, &out_of_range);

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
  float high_weight_x, high_weight_y, high_weight_z;
  int out_of_range = 0;
  device_interpolate_get_coordinate_weight(coordinate_x, model_x, &low_x, &low_weight_x, &high_weight_x, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_y, model_y, &low_y, &low_weight_y, &high_weight_y, &out_of_range);
  device_interpolate_get_coordinate_weight(coordinate_z, model_z, &low_z, &low_weight_z, &high_weight_z, &out_of_range);

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
	  
	  atomicAdd(&model[model_x*model_y*index_z + model_x*index_y + index_x],
		    weight_x*weight_y*weight_z*value_weight*value);
	  atomicAdd(&model_weights[model_x*model_y*index_z + model_x*index_y + index_x],
		    weight_x*weight_y*weight_z*value_weight);
	}
      }
    }
  }
}

__device__ void device_model_set_nn(float *const model, float *const model_weights,
				    const int model_x, const int model_y, const int model_z,
				    const float coordinate_x, const float coordinate_y, const float coordinate_z,
				    const float value, const float value_weight)
{
  int index_x = (int) (coordinate_x + 0.5);
  int index_y = (int) (coordinate_y + 0.5);
  int index_z = (int) (coordinate_z + 0.5);
  if (index_x >= 0 && index_x < model_x &&
      index_y >= 0 && index_y < model_y &&
      index_z >= 0 && index_z < model_y) {
    atomicAdd(&model[model_x*model_y*index_z + model_x*index_y + index_x],
	      value_weight*value);
    atomicAdd(&model_weights[model_x*model_y*index_z + model_x*index_y + index_x],
	      value_weight);
  }
}

__device__ void device_insert_slice(float *const model, float *const model_weights,
				    const int model_x, const int model_y, const int model_z,
				    const float *const slice, const int image_x, const int image_y,
				    const float slice_weight,
				    const float *const rotation, const float *const coordinates,
				    const int interpolation)
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
      if (slice[y*image_x+x] >= 0.) {
	/* This is just a matrix multiplication with rotation */
	new_x = m00*coordinates_0[y*image_x+x] + m01*coordinates_1[y*image_x+x] + m02*coordinates_2[y*image_x+x] + model_x/2.0 - 0.5;
	new_y = m10*coordinates_0[y*image_x+x] + m11*coordinates_1[y*image_x+x] + m12*coordinates_2[y*image_x+x] + model_y/2.0 - 0.5;
	new_z = m20*coordinates_0[y*image_x+x] + m21*coordinates_1[y*image_x+x] + m22*coordinates_2[y*image_x+x] + model_z/2.0 - 0.5;

	if (interpolation == 0) {
	  device_model_set_nn(model, model_weights, model_x, model_y, model_z, new_x, new_y, new_z, slice[y*image_x+x], slice_weight);
	} else {
	  device_model_set(model, model_weights, model_x, model_y, model_z, new_x, new_y, new_z, slice[y*image_x+x], slice_weight);
	}
      }
    }
  }
}


__global__ void kernel_insert_slices(float *const model, float *const model_weights,
				     const int model_x, const int model_y, const int model_z,
				     const float *const slices, const int image_x, const int image_y,
				     const float *const slice_weights, const float *const rotations,
				     const float *const coordinates, const int interpolation) {
  const int rotation_index = blockIdx.x;
  device_insert_slice(model, model_weights, model_x, model_y, model_z,
		      &slices[image_x*image_y*rotation_index], image_x, image_y,
		      slice_weights[rotation_index], &rotations[4*rotation_index],
		      coordinates, interpolation);
}

void cuda_insert_slices(float *const model, float *const model_weights,
			const int model_x, const int model_y, const int model_z,
			const float *const slices, const int image_x, const int image_y,
			const float *const slice_weights,
			const float *const rotations, const int number_of_rotations,
			const float *const coordinates, const int interpolation)
{
  int nblocks = number_of_rotations;
  int nthreads = NTHREADS;
  kernel_insert_slices<<<nblocks, nthreads>>>(model, model_weights, model_x, model_y, model_z,
					      slices, image_x, image_y, slice_weights, rotations,
					      coordinates, interpolation);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__device__ void device_insert_slice_partial(float *const model, float *const model_weights,
					    const int model_x_tot, const int model_x_min, const int model_x_max,
					    const int model_y_tot, const int model_y_min, const int model_y_max,
					    const int model_z_tot, const int model_z_min, const int model_z_max,
					    const float *const slice, const int image_x, const int image_y,
					    const float slice_weight,
					    const float *const rotation, const float *const coordinates,
					    const int interpolation)
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
      if (slice[y*image_x+x] >= 0.) {
	/* This is just a matrix multiplication with rotation */
	new_x = m00*coordinates_0[y*image_x+x] + m01*coordinates_1[y*image_x+x] + m02*coordinates_2[y*image_x+x] + model_x_tot/2.0 - 0.5;
	new_y = m10*coordinates_0[y*image_x+x] + m11*coordinates_1[y*image_x+x] + m12*coordinates_2[y*image_x+x] + model_y_tot/2.0 - 0.5;
	new_z = m20*coordinates_0[y*image_x+x] + m21*coordinates_1[y*image_x+x] + m22*coordinates_2[y*image_x+x] + model_z_tot/2.0 - 0.5;

	if (interpolation == 0) {
	  device_model_set_nn(model, model_weights, model_x_max-model_x_min, model_y_max-model_y_min, model_z_max-model_z_min,
			      new_x-(float)model_x_min, new_y-(float)model_y_min, new_z-(float)model_z_min, slice[y*image_x+x], slice_weight);
	} else {
	  device_model_set(model, model_weights, model_x_max-model_x_min, model_y_max-model_y_min, model_z_max-model_z_min,
			   new_x-(float)model_x_min, new_y-(float)model_y_min, new_z-(float)model_z_min, slice[y*image_x+x], slice_weight);
	}
      }
    }
  }
}

__global__ void kernel_insert_slices_partial(float *const model, float *const model_weights,
					     const int model_x_tot, const int model_x_min, const int model_x_max,
					     const int model_y_tot, const int model_y_min, const int model_y_max,
					     const int model_z_tot, const int model_z_min, const int model_z_max,
					     const float *const slices, const int image_x, const int image_y,
					     const float *const slice_weights, const float *const rotations,
					     const float *const coordinates, const int interpolation) {
  const int rotation_index = blockIdx.x;
  device_insert_slice_partial(model, model_weights, model_x_tot, model_x_min, model_x_max,
			      model_y_tot, model_y_min, model_y_max, model_z_tot, model_z_min, model_z_max,
			      &slices[image_x*image_y*rotation_index], image_x, image_y,
			      slice_weights[rotation_index], &rotations[4*rotation_index],
			      coordinates, interpolation);
}


void cuda_insert_slices_partial(float *const model, float *const model_weights,
				const int model_x_tot, const int model_x_min, const int model_x_max,
				const int model_y_tot, const int model_y_min, const int model_y_max,
				const int model_z_tot, const int model_z_min, const int model_z_max,
				const float *const slices, const int image_x, const int image_y,
				const float *const slice_weights,
				const float *const rotations, const int number_of_rotations,
				const float *const coordinates, const int interpolation)
{
  int nblocks = number_of_rotations;
  int nthreads = NTHREADS;
  kernel_insert_slices_partial<<<nblocks, nthreads>>>(model, model_weights, model_x_tot, model_x_min, model_x_max,
						      model_y_tot, model_y_min, model_y_max, model_z_tot, model_z_min, model_z_max,
						      slices, image_x, image_y, slice_weights, rotations, coordinates, interpolation);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_update_slices(float *const slices, const float *const patterns,
				     const int number_of_patterns, const int number_of_pixels,
				     const float *const responsabilities)
{
  const int index_rotation = blockIdx.x;
  float sum;
  float weight;
  for (int pixel_index = threadIdx.x; pixel_index < number_of_pixels; pixel_index += blockDim.x) {
    sum = 0.;
    weight = 0.;
    for (int pattern_index = 0; pattern_index < number_of_patterns; pattern_index++) {
      if (patterns[pattern_index*number_of_pixels + pixel_index] >= 0.) {
	sum += patterns[pattern_index*number_of_pixels + pixel_index] * responsabilities[index_rotation*number_of_patterns + pattern_index];
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

void cuda_update_slices(float *const slices, const int number_of_rotations,
			const float *const patterns, const int number_of_patterns,
			const int image_x, const int image_y,
			const float *const responsabilities)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices<<<nblocks, nthreads>>>(slices, patterns, number_of_patterns, image_x*image_y, responsabilities);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_update_slices_scaling(float *const slices, const float *const patterns,
					     const int number_of_patterns, const int number_of_pixels,
					     const float *const responsabilities, const float *const scaling)
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

void cuda_update_slices_scaling(float *const slices, const int number_of_rotations,
				const float *const patterns, const int number_of_patterns,
				const int image_x, const int image_y,
				const float *const responsabilities, const float *const scaling)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices_scaling<<<nblocks, nthreads>>>(slices, patterns, number_of_patterns, image_x*image_y, responsabilities, scaling);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

/* This can't handle masks att the moment. Need to think about how to handle masked out data in the sparse implemepntation
 */
__global__ void kernel_update_slices_sparse(float *const slices, const int number_of_pixels,
					    const int *const pattern_start_indices, const int *const pattern_indices,
					    const float *const pattern_values, const int number_of_patterns,
					    const float *const responsabilities)
{
  //const int number_of_rotations = gridDim.x;
  const int index_rotation = blockIdx.x;

  int index_pixel;

  for (int index_pattern = threadIdx.x; index_pattern < number_of_patterns; index_pattern += blockDim.x) {
    for (int value_index = pattern_start_indices[index_pattern]; value_index < pattern_start_indices[index_pattern+1]; value_index += blockDim.x) {
      index_pixel = pattern_indices[value_index];
      atomicAdd(&slices[index_rotation*number_of_pixels + index_pixel],
		pattern_values[value_index] * responsabilities[index_rotation*number_of_patterns + index_pattern]);
    }
  }

  float normalization_factor;  
  for (int index_pattern = threadIdx.x; index_pattern < number_of_patterns; index_pattern += blockDim.x) {
    normalization_factor += responsabilities[index_rotation*number_of_patterns + index_pattern];
  }
  for (int index_pattern = threadIdx.x; index_pattern < number_of_patterns; index_pattern += blockDim.x) {
    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      slices[index_rotation*number_of_pixels + index_pixel] *= normalization_factor;
    }
  }
}

void cuda_update_slices_sparse(float *const slices, const int number_of_rotations, const int *const pattern_start_indices,
			       const int *const pattern_indices, const float *const pattern_values, const int number_of_patterns,
			       const int image_x, const int image_y, const float *const responsabilities)
{
  const int nblocks = number_of_rotations;
  const int nthreads = NTHREADS;
  kernel_update_slices_sparse<<<nblocks, nthreads>>>(slices, image_x*image_y, pattern_start_indices,
						     pattern_indices, pattern_values, number_of_patterns,
						     responsabilities);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__device__ float device_compare_balanced(const float expected_value, const float measured_value) 
{
  return pow(expected_value - measured_value, 2) / expected_value;
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

void cuda_calculate_responsabilities(const float *const patterns, const int number_of_patterns, const float *const slices, const int number_of_rotations,
				     const int image_x, const int image_y, float *const responsabilities, const float sigma)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_responsabilities<<<nblocks, nthreads>>>(patterns, slices, image_x*image_y, responsabilities, sigma);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_responsabilities_poisson(const float* const patterns, const float *const slices, const int number_of_pixels, float *const responsabilities, const float *const log_factorial_table)
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
      sum += (-slice[index]) + ((int) pattern[index]) * logf(slice[index]) - log_factorial_table[(int) pattern[index]];
    }
  }
  sum_cache[threadIdx.x] = sum;

  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = sum_cache[0];
  }
}

void cuda_calculate_responsabilities_poisson(const float *const patterns, const int number_of_patterns,
					     const float *const slices, const int number_of_rotations,
					     const int image_x, const int image_y,
					     float *const responsabilities, const float *const log_factorial_table)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_responsabilities_poisson<<<nblocks, nthreads>>>(patterns, slices, image_x*image_y, responsabilities, log_factorial_table);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_responsabilities_poisson_scaling(const float *const patterns, const float *const slices, const int number_of_pixels,
								  const float *const scalings, float *const responsabilities, const float *const log_factorial_table)
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
      sum += (-slice[index]/scaling) + ((int) pattern[index]) * logf(slice[index]/scaling) - log_factorial_table[(int) pattern[index]];
    }
  }
  sum_cache[threadIdx.x] = sum;

  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = sum_cache[0];
  }
}

void cuda_calculate_responsabilities_poisson_scaling(const float *const patterns, const int number_of_patterns,
						     const float *const slices, const int number_of_rotations,
						     const int image_x, const int image_y, const float *const scalings,
						     float *const responsabilities, const float *const log_factorial_table)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_responsabilities_poisson_scaling<<<nblocks, nthreads>>>(patterns, slices, image_x*image_y, scalings, responsabilities, log_factorial_table);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_scaling_poisson(const float *const patterns, const float *const slices, float *const scaling, const int number_of_pixels) {
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
    //printf("scaling[%d, %d] = %g / %g\n", index_slice, index_pattern, sum_slice_cache[0], sum_pattern_cache[0]);
    scaling[index_slice*number_of_patterns + index_pattern] = sum_slice_cache[0] / sum_pattern_cache[0];
  }
}

void cuda_calculate_scaling_poisson(const float *const patterns, const int number_of_patterns,
				    const float *const slices, const int number_of_rotations,
				    const int number_of_pixels, float *const scaling)
{
  dim3 nblocks(number_of_patterns, number_of_rotations);
  int nthreads = NTHREADS;
  kernel_calculate_scaling_poisson<<<nblocks, nthreads>>>(patterns, slices, scaling, number_of_pixels);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}

__global__ void kernel_sum_slices(const float *const slices, const int number_of_pixels, float *const slice_sums)
{
  __shared__ float sum_cache[NTHREADS];
  
  const int index_slice = blockIdx.x;
  const float *const slice = &slices[number_of_pixels*index_slice];

  float sum;
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

__global__ void kernel_calculate_responsabilities_sparse(const int *const pattern_start_indices, const int *const pattern_indices,
							 const float *const pattern_values,
							 const float *const slices, const int number_of_pixels,
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
  for (int index = pattern_start_indices[index_pattern]+threadIdx.x; index < pattern_start_indices[index_pattern+1]; index += blockDim.x) {
    index_pixel = pattern_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += ((int) pattern_values[index]) * logf(slice[index_pixel]) - log_factorial_table[(int)pattern_values[index]];
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice] + sum_cache[0];
  }
}

/* Need to calculate slice sums before calling this kernel. But it can be done in python */
void cuda_calculate_responsabilities_sparse(const int *const pattern_start_indices, const int *const pattern_indices,
					    const float *const pattern_values, const int number_of_patterns,
					    const float *const slices, const int number_of_rotations, const int image_x, const int image_y,
					    float *const responsabilities, float *const slice_sums,
					    const float *const log_factorial_table)
{
  const int nblocks_sum_slices = number_of_rotations;
  const int nthreads_sum_slices = NTHREADS;
  kernel_sum_slices<<<nblocks_sum_slices, nthreads_sum_slices>>>(slices, image_x*image_y, slice_sums);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
  
  const dim3 nblocks_calc_resp(number_of_patterns, number_of_rotations);
  const int nthreads_calc_resp = NTHREADS;
  kernel_calculate_responsabilities_sparse<<<nblocks_calc_resp, nthreads_calc_resp>>>(pattern_start_indices, pattern_indices, pattern_values,
										      slices, image_x*image_y, responsabilities,
										      slice_sums, log_factorial_table);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}


__global__ void kernel_rotate_model(const float *const model, float *const rotated_model, const int model_x,
				    const int model_y, const int model_z, const float *const rotation) {

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
    new_x = model_x/2. - 0.5 + (rotation_matrix[0]*start_z +
				rotation_matrix[1]*start_y +
				rotation_matrix[2]*start_x);
    new_y = model_y/2. - 0.5 + (rotation_matrix[3]*start_z +
				rotation_matrix[4]*start_y +
				rotation_matrix[5]*start_x);
    new_z = model_z/2. - 0.5 + (rotation_matrix[6]*start_z +
				rotation_matrix[7]*start_y +
				rotation_matrix[8]*start_x);
    rotated_model[index] = device_model_get(model, model_x, model_y, model_z, new_x, new_y, new_z);
    //rotated_model[index] = start_x + 100.*start_y + 100000.*start_z;
  }
}

void cuda_rotate_model(const float *const model, float *const rotated_model, const int model_x,
		       const int model_y, const int model_z, const float *const rotation)
{
  const int nthreads = NTHREADS;
  const int nblocks = (model_x*model_y*model_z-1) / nthreads + 1;
  kernel_rotate_model<<<nblocks, nthreads>>>(model, rotated_model, model_x, model_y, model_z, rotation);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());
}
