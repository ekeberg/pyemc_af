
//void cuda_add_one(float *const array, const int length);
void set_to_value(float *const array, const int size, const float value);
void masked_set(float *const array, const int *const mask, const int size, const float value);

float *int_to_float_pointer(const unsigned long long pointer_int);
int *int_to_int_pointer(const unsigned long long pointer_int);

void cuda_expand_model(const float *const model, const int model_x, const int model_y, const int model_z,
		       float *const slices, const int image_x, const int image_y,
		       const float *const rotations, const int number_of_rotations,
		       const float *const coordinates);

void cuda_insert_slices(float *const model, float *const model_weights,
			const int model_x, const int model_y, const int model_z,
			const float *const slices, const int image_x, const int image_y,
			const float *const slice_weights,
			const float *const rotations, const int number_of_rotations,
			const float *const coordinates);

void cuda_update_slices(float *const slices, const int number_of_rotations, const float *const patterns, const int number_of_patterns,
			const int image_x, const int image_y, const float *const responsabilities);

void cuda_calculate_responsabilities(const float *const patterns, const int number_of_patterns, const float *const slices, const int number_of_rotations,
				     const int image_x, const int image_y, float *const responsabilities, const float sigma);

void cuda_calculate_responsabilities_sparse(const int *const pattern_start_indices, const int *const pattern_indices,
					    const float *const pattern_values, const int number_of_patterns,
					    const float *const slices, const int number_of_rotations, const int image_x, const int image_y,
					    float *const responsabilities, float *const slice_sums,
					    const float *const log_factorial_table);

void cuda_calculate_responsabilities_poisson(const float *const patterns, const int number_of_patterns,
					     const float *const slices, const int number_of_rotations,
					     const int image_x, const int image_y,
					     float *const responsabilities, const float *const log_factorial_table);

void cuda_calculate_responsabilities_poisson_scaling(const float *const patterns, const int number_of_patterns,
						     const float *const slices, const int number_of_rotations,
						     const int image_x, const int image_y, const float *const scalings,
						     float *const responsabilities, const float *const log_factorial_table);

void cuda_calculate_scaling_poisson(const float *const patterns, const int number_of_patterns,
				    const float *const slices, const int number_of_rotations,
				    const int number_of_pixels, float *const scalings);

void cuda_update_slices_sparse(float *const slices, const int number_of_rotations, const int *const pattern_start_indices,
			       const int *const pattern_indices, const float *const pattern_values, const int number_of_patterns,
			       const int image_x, const int image_y, const float *const responsabilities);

void cuda_rotate_model(const float *const model, float *const rotated_model, const int model_x,
		       const int model_y, const int model_z, const float *const rotation);

