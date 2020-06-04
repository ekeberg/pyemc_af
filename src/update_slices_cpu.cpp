#include <Python.h>
#include <emc_cpu.h>

using namespace std;

void update_slices(float *const slices,
		   const int number_of_rotations,
		   const float *const patterns,
		   const int number_of_patterns,
		   const int image_x,
		   const int image_y,
		   const float *const responsabilities)
{
  float sum;
  float weight;
  const int number_of_pixels = image_x*image_y;
  for (int index_rotation = 0; index_rotation < number_of_rotations; index_rotation++) {
    for (int pixel_index = 0; pixel_index < number_of_pixels; pixel_index++) {
      sum = 0.;
      weight = 0.;
      for (int pattern_index = 0; pattern_index < number_of_patterns; pattern_index++) {
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
  float sum;
  float weight;
  const int number_of_pixels = image_x*image_y;
  for (int index_rotation = 0; index_rotation < number_of_rotations; index_rotation++) {
    for (int pixel_index = 0; pixel_index < number_of_pixels; pixel_index++) {
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
  float sum;
  float weight;
  const int number_of_pixels = image_x*image_y;
  for (int index_rotation = 0; index_rotation < number_of_rotations; index_rotation++) {
    for (int pixel_index = 0; pixel_index < number_of_pixels; pixel_index++) {
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
}

/* This can't handle masks att the moment. Need to think about how to handle masked out data in the sparse implemepntation
 */
void update_slices_sparse(float *const slices,
			  const int number_of_rotations,
			  const int *const pattern_start_indices,
			  const int *const pattern_indices,
			  const float *const pattern_values,
			  const int number_of_patterns,
			  const int image_x,
			  const int image_y,
			  const float *const responsabilities)
{
  int index_pixel;
  float normalization_factor;
  const int number_of_pixels = image_x*image_y;
  for (int index_rotation = 0; index_rotation < number_of_rotations; index_rotation++) {
    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      slices[index_rotation*number_of_pixels + index_pixel] = 0.;
    }
    for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
      for (int value_index = pattern_start_indices[index_pattern];
	   value_index < pattern_start_indices[index_pattern+1];
	   value_index += 1) {
	index_pixel = pattern_indices[value_index];
	slices[index_rotation*number_of_pixels + index_pixel] += (pattern_values[value_index] *
								  responsabilities[index_rotation*number_of_patterns + index_pattern]);
      }
    }

    normalization_factor = 0.;
    for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
      normalization_factor += responsabilities[index_rotation*number_of_patterns + index_pattern];
    }

    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      slices[index_rotation*number_of_pixels + index_pixel] *= 1./normalization_factor;
    }
  }
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
				  const float *const scaling)
{
  int index_pixel;
  float normalization_factor;
  const int number_of_pixels = image_x*image_y;
  for (int index_rotation = 0; index_rotation < number_of_rotations; index_rotation++) {
    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      slices[index_rotation*number_of_pixels + index_pixel] = 0.;
    }

    for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
      for (int value_index = pattern_start_indices[index_pattern];
	   value_index < pattern_start_indices[index_pattern+1];
	   value_index += 1) {
	index_pixel = pattern_indices[value_index];
	slices[index_rotation*number_of_pixels + index_pixel] += (pattern_values[value_index] *
								  scaling[index_rotation*number_of_patterns + index_pattern] *
								  responsabilities[index_rotation*number_of_patterns + index_pattern]);
      }
    }

    normalization_factor = 0.;
    for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
      normalization_factor += responsabilities[index_rotation*number_of_patterns + index_pattern];
    }
    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      slices[index_rotation*number_of_pixels + index_pixel] *= 1./normalization_factor;
    }

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
  int index_pixel;
  float normalization_factor;
  const int number_of_pixels = image_x*image_y;
  for (int index_rotation = 0; index_rotation < number_of_rotations; index_rotation++) {
    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      slices[index_rotation*number_of_pixels + index_pixel] = 0.;
    }

    for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
      for (int value_index = pattern_start_indices[index_pattern];
	   value_index < pattern_start_indices[index_pattern+1];
	   value_index += 1) {
	index_pixel = pattern_indices[value_index];
	slices[index_rotation*number_of_pixels + index_pixel] += (pattern_values[value_index] *
								  scaling[index_pattern] *
								  responsabilities[index_rotation*number_of_patterns + index_pattern]);
      }
    }

    normalization_factor = 0.;
    for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
      normalization_factor += responsabilities[index_rotation*number_of_patterns + index_pattern];
    }
    for (int index_pixel = 0; index_pixel < number_of_pixels; index_pixel++) {
      slices[index_rotation*number_of_pixels + index_pixel] *= 1./normalization_factor;
    }

  }
}
