
//void cuda_add_one(float *const array, const int length);
void set_to_value(float *const array,
		  const int size,
		  const float value);
void masked_set(float *const array,
		const int *const mask,
		const int size,
		const float value);

float *int_to_float_pointer(const unsigned long long pointer_int);
int *int_to_int_pointer(const unsigned long long pointer_int);

void expand_model(const float *const model,
		  const int model_x,
		  const int model_y,
		  const int model_z,
		  float *const slices,
		  const int image_x,
		  const int image_y,
		  const float *const rotations,
		  const int number_of_rotations,
		  const float *const coordinates);

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
		   const int interpolation);

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
			   const int interpolation);

void update_slices(float *const slices,
		   const int number_of_rotations,
		   const float *const patterns,
		   const int number_of_patterns,
		   const int image_x,
		   const int image_y,
		   const float *const responsabilities);

void update_slices_scaling(float *const slices,
			   const int number_of_rotations,
			   const float *const patterns,
			   const int number_of_patterns,
			   const int image_x,
			   const int image_y,
			   const float *const responsabilities,
			   const float *const scaling);

void update_slices_per_pattern_scaling(float *const slices,
				       const int number_of_rotations,
				       const float *const patterns,
				       const int number_of_patterns,
				       const int image_x,
				       const int image_y,
				       const float *const responsabilities,
				       const float *const scaling);

void update_slices_sparse(float *const slices,
			  const int number_of_rotations,
			  const int *const pattern_start_indices,
			  const int *const pattern_indices,
			  const float *const pattern_values,
			  const int number_of_patterns,
			  const int image_x,
			  const int image_y,
			  const float *const responsabilities,
			  const float resp_threshold);

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
				  const float *const scaling);

void update_slices_sparse_per_pattern_scaling(float *const slices,
					      const int number_of_rotations,
					      const int *const pattern_start_indices,
					      const int *const pattern_indices,
					      const float *const pattern_values,
					      const int number_of_patterns,
					      const int image_x,
					      const int image_y,
					      const float *const responsabilities,
					      const float *const scaling);

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
				       const float *const log_factorial_table);

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
					       const float *const log_factorial_table);

void calculate_responsabilities_sparse_per_pattern_scaling(const int *const pattern_start_indices,
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
							   const float *const log_factorial_table);

void calculate_responsabilities_poisson(const float *const patterns,
					const int number_of_patterns,
					const float *const slices,
					const int number_of_rotations,
					const int image_x,
					const int image_y,
					float *const responsabilities,
					const float *const log_factorial_table);

void calculate_responsabilities_poisson_scaling(const float *const patterns,
						const int number_of_patterns,
						const float *const slices,
						const int number_of_rotations,
						const int image_x,
						const int image_y,
						const float *const scalings,
						float *const responsabilities,
						const float *const log_factorial_table);

void calculate_responsabilities_poisson_per_pattern_scaling(const float *const patterns,
							    const int number_of_patterns,
							    const float *const slices,
							    const int number_of_rotations,
							    const int image_x,
							    const int image_y,
							    const float *const scalings,
							    float *const responsabilities,
							    const float *const log_factorial_table);

void calculate_scaling_poisson(const float *const patterns,
			       const int number_of_patterns,
			       const float *const slices,
			       const int number_of_rotations,
			       const int number_of_pixels,
			       float *const scalings);

void calculate_scaling_poisson_sparse(const int *const pattern_start_indices,
				      const int *const pattern_indices,
				      const float *const pattern_values,
				      const int number_of_patterns,
				      const float *const slices,
				      const int number_of_rotations,
				      const int number_of_pixels,
				      float *const scaling);

void calculate_scaling_per_pattern_poisson(const float *const patterns,
					   const int number_of_patterns,
					   const float *const slices,
					   const int number_of_rotations,
					   const int number_of_pixels,
					   const float *const responsabilities,
					   float *const scalings);

void calculate_scaling_per_pattern_poisson_sparse(const int *const pattern_start_indices,
						  const int *const pattern_indices,
						  const float *const pattern_values,
						  const int number_of_patterns,
						  const float *const slices,						  
						  const int number_of_rotations,
						  const int number_of_pixels,
						  const float *const responsabilities,
						  float *const scaling);

void rotate_model(const float *const model,
		  float *const rotated_model,
		  const int model_x,
		  const int model_y,
		  const int model_z,
		  const float *const rotation);

