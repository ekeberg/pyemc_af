#include <Python.h>
#include <emc_cpu.h>	

using namespace std;

void calculate_responsabilities(const float *const patterns,
				const int number_of_patterns,
				const float *const slices,
				const int number_of_rotations,
				const int image_x,
				const int image_y,
				float *const responsabilities,
				const float sigma)
{
  const int number_of_pixels = image_x*image_y;
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const pattern = &patterns[number_of_pixels*index_pattern];
      const float *const slice = &slices[number_of_pixels*index_slice];
  
      /* Use a gaussian with a sqrt normalization */
      float sum = 0.;
      float weight = 0.;
      for (int index = 0; index < number_of_pixels; index++) {
	if (pattern[index] >= 0. && slice[index] > 0.) {
	  sum += pow((slice[index] - pattern[index]) / sqrt(slice[index]), 2);
	  weight += 1.;
	}
      }

      responsabilities[index_slice*number_of_patterns + index_pattern] = -sum/(2.*weight*pow(sigma, 2));
    }
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
  const int number_of_pixels = image_x*image_y;
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const pattern = &patterns[number_of_pixels*index_pattern];
      const float *const slice = &slices[number_of_pixels*index_slice];
  
      /* Use a gaussian with a sqrt normalization */
      float sum = 0.;
      for (int index = 0; index < number_of_pixels; index++) {
	if (pattern[index] >= 0. && slice[index] > 0.) {
	  sum += ((-slice[index]) +
		  ((int) pattern[index]) * logf(slice[index]) -
		  log_factorial_table[(int) pattern[index]]);
	}
      }
      responsabilities[index_slice*number_of_patterns + index_pattern] = sum;
    }
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
  const int number_of_pixels = image_x*image_y;
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const pattern = &patterns[number_of_pixels*index_pattern];
      const float *const slice = &slices[number_of_pixels*index_slice];
      const float scaling = scalings[index_slice*number_of_patterns + index_pattern];
  
      // Use a gaussian with a sqrt normalization
      float sum = 0.;
      for (int index = 0; index < number_of_pixels; index++) {
	if (pattern[index] >= 0. && slice[index] > 0.) {
	  sum += ((-slice[index]/scaling) +
		  ((int) pattern[index]) * logf(slice[index]/scaling) -
		  log_factorial_table[(int) pattern[index]]);
	}
      }

      responsabilities[index_slice*number_of_patterns + index_pattern] = sum;
    }
  }
}

void kernel_sum_slices(const float *const slices,
		       const int number_of_pixels,
		       const int number_of_rotations,
		       float *const slice_sums)
{
  for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
    const float *const slice = &slices[number_of_pixels*index_slice];

    float sum = 0.;
    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      if (slice[index_pixel] > 0.) {
	sum += slice[index_pixel];
      }
    }
    slice_sums[index_slice] = sum;
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
  kernel_sum_slices(slices,
		    image_x*image_y,
		    number_of_rotations,
		    slice_sums);

  int index_pixel;
  const int number_of_pixels = image_x*image_y;
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const slice = &slices[number_of_pixels*index_slice];
  
      float sum = 0.;
      for (int index = pattern_start_indices[index_pattern];
	   index < pattern_start_indices[index_pattern+1];
	   index++) {
	index_pixel = pattern_indices[index];
	if (slice[index_pixel] > 0.) {
	  sum += (((int) pattern_values[index]) *
		  logf(slice[index_pixel]) -
		  log_factorial_table[(int)pattern_values[index]]);
	}
      }
      responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice] + sum;
    }
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
  kernel_sum_slices(slices,
		    image_x*image_y,
		    number_of_rotations,
		    slice_sums);
  int index_pixel;
  float sum;
  const int number_of_pixels = image_x*image_y;
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const slice = &slices[number_of_pixels*index_slice];
      const float this_scaling = scaling[index_slice*number_of_patterns + index_pattern];

      sum = 0.;
      for (int index = pattern_start_indices[index_pattern];
	   index < pattern_start_indices[index_pattern+1];
	   index++) {
	index_pixel = pattern_indices[index];
	if (slice[index_pixel] > 0.) {
	  sum += ((int) pattern_values[index]) * logf(slice[index_pixel]/this_scaling) - log_factorial_table[(int)pattern_values[index]];
	}
      }
      responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice]/this_scaling + sum;
    }
  }
}

