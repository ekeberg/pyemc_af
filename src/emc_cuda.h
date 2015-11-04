
//void cuda_add_one(float *const array, const int length);
float *int_to_float_pointer(const unsigned long long pointer_int);
int *int_to_int_pointer(const unsigned long long pointer_int);


void cuda_add_one(float *const array, const int length);


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
