#include <Python.h>
#include <emc_cpu.h>

using namespace std;

void calculate_scaling_poisson(const float *const patterns,
			       const int number_of_patterns,
			       const float *const slices,
			       const int number_of_rotations,
			       const int number_of_pixels,
			       float *const scaling)
{
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const pattern = &patterns[number_of_pixels*index_pattern];
      const float *const slice = &slices[number_of_pixels*index_slice];

      float sum_slice = 0.;
      float sum_pattern = 0.;
      for (int index = 0; index < number_of_pixels; index++) {
	if (pattern[index] >= 0. && slice[index] >= 0.) {
	  sum_slice += slice[index];
	  sum_pattern += pattern[index];
	}
      }

      scaling[index_slice*number_of_patterns + index_pattern] = sum_slice / sum_pattern;
    }
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
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      //const float *const pattern = &patterns[number_of_pixels*index_pattern];
      const float *const slice = &slices[number_of_pixels*index_slice];

      const int this_start_index = pattern_start_indices[index_pattern];
      const int this_end_index = pattern_start_indices[index_pattern+1];

      float sum_slice = 0.;
      float sum_pattern = 0.;

      for (int index = this_start_index; index < this_end_index; index++) {
	if (slice[pattern_indices[index]] >= 0.) {
	  sum_pattern += pattern_values[index];
	}
      }

      for (int index = 0; index < number_of_pixels; index++) {
	if (slice[index] >= 0.) {
	  sum_slice += slice[index];
	}
      }

      scaling[index_slice*number_of_patterns + index_pattern] = sum_slice / sum_pattern;
    }
  }
}

void calculate_scaling_per_pattern_poisson(const float *const patterns,
					   const int number_of_patterns,
					   const float *const slices,
					   const int number_of_rotations,
					   const int number_of_pixels,
					   const float *const responsabilities,
					   float *const scaling)
{
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    const float *const pattern = &patterns[number_of_pixels*index_pattern];
    float sum_nominator = 0.;
    float sum_denominator = 0.;
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const slice = &slices[number_of_pixels*index_slice];
      for (int index = 0; index < number_of_pixels; index++) {
	if (pattern[index] >= 0. && slice[index] >= 0.) {
	  sum_nominator += responsabilities[index_slice*number_of_patterns + index_pattern] * slice[index];
	  sum_denominator += responsabilities[index_slice*number_of_patterns + index_pattern] * pattern[index];
	}
      }      
    }
    scaling[index_pattern] = sum_nominator / sum_denominator;
  }
}

void calculate_scaling_per_pattern_poisson_sparse(const int *const pattern_start_indices,
						  const int *const pattern_indices,
						  const float *const pattern_values,
						  const int number_of_patterns,
						  const float *const slices,
						  const int number_of_rotations,
						  const int number_of_pixels,
						  const float *const responsabilities,
						  float *const scaling)
{
  for (int index_pattern = 0; index_pattern < number_of_patterns; index_pattern++) {
    const int this_start_index = pattern_start_indices[index_pattern];
    const int this_end_index = pattern_start_indices[index_pattern+1];

    float sum_nominator = 0.;
    float sum_denominator = 0.;
    for (int index_slice = 0; index_slice < number_of_rotations; index_slice++) {
      const float *const slice = &slices[number_of_pixels*index_slice];
      for (int index = this_start_index; index < this_end_index; index++) {
	if (slice[pattern_indices[index]] >= 0) {
	  sum_nominator += responsabilities[index_slice*number_of_patterns + index_pattern] * slice[pattern_indices[index]];
	  sum_denominator += responsabilities[index_slice*number_of_patterns + index_pattern] * pattern_values[index];
	}
      }
    }
    scaling[index_pattern] = sum_nominator / sum_denominator;
  }
}
