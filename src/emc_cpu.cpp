#include <Python.h>
#include <emc_cpu.h>

using namespace std;

void set_to_value(float *const array,
		  const int size,
		  const float value)
{
  for (int index = 0; index < size; index++) {
    array[index] = value;
  }
}

void masked_set(float *const array,
		const int *const mask,
		const int size,
		const float value)
{
  for (int index = 0; index < size; index++) {
    if (mask[index] > 0) {
      array[index] = value;
    }
  }
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

/* could be shared with cuda (as __device__) */
void device_interpolate_get_coordinate_weight(const float coordinate,
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

/* could be shared with cuda (as __device__) */
float device_model_get(const float *const model,
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

/* could maybe be shared with cuda (as __device__) but the inner loop is different*/
void device_get_slice(const float *const model,
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
    for (int y = 0; y < image_y; y++) {
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
  for (int rotation_index = 0; rotation_index < number_of_rotations; rotation_index++) {
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
}

/* could be shared with cuda (as __device__) */
void device_model_set(float *const model,
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

	  model[model_x*model_z*index_x + model_z*index_y + index_z] += weight_x*weight_y*weight_z*value_weight*value;
	  model_weights[model_z*model_y*index_x + model_z*index_y + index_z] += weight_x*weight_y*weight_z*value_weight;
	}
      }
    }
  }
}

void device_model_set_nn(float *const model,
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
    model[model_z*model_y*index_x + model_z*index_y + index_z] += value_weight*value;
    model_weights[model_z*model_y*index_x + model_z*index_y + index_z] += value_weight;
  }
}

void device_insert_slice(float *const model,
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
    for (int y = 0; y < image_y; y++) {
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
  for (int rotation_index = 0; rotation_index < number_of_rotations; rotation_index++) {
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
}

void rotate_model(const float *const model,
		  float *const rotated_model,
		  const int model_x,
		  const int model_y,
		  const int model_z,
		  const float *const rotation)
{
  for (int index = 0; index < model_x*model_y*model_z; index++) {
    
    float rotation_matrix_00 = rotation[0]*rotation[0] + rotation[1]*rotation[1] - rotation[2]*rotation[2] - rotation[3]*rotation[3]; // 00
    float rotation_matrix_01 = 2.0f*rotation[1]*rotation[2] - 2.0f*rotation[0]*rotation[3]; // 01
    float rotation_matrix_02 = 2.0f*rotation[1]*rotation[3] + 2.0f*rotation[0]*rotation[2]; // 02
    float rotation_matrix_10 = 2.0f*rotation[1]*rotation[2] + 2.0f*rotation[0]*rotation[3]; // 10
    float rotation_matrix_11 = rotation[0]*rotation[0] - rotation[1]*rotation[1] + rotation[2]*rotation[2] - rotation[3]*rotation[3]; // 11
    float rotation_matrix_12 = 2.0f*rotation[2]*rotation[3] - 2.0f*rotation[0]*rotation[1]; // 12
    float rotation_matrix_20 = 2.0f*rotation[1]*rotation[3] - 2.0f*rotation[0]*rotation[2]; // 20
    float rotation_matrix_21 = 2.0f*rotation[2]*rotation[3] + 2.0f*rotation[0]*rotation[1]; // 21
    float rotation_matrix_22 = rotation[0]*rotation[0] - rotation[1]*rotation[1] - rotation[2]*rotation[2] + rotation[3]*rotation[3]; // 22

    float start_z = ((float) ((index % (model_x*model_y)) % model_x)) - model_x/2. + 0.5;
    float start_y = ((float) ((index / model_x) % model_y)) - model_y/2. + 0.5;
    float start_x = ((float) (index / (model_x*model_y))) - model_z/2. + 0.5;

    /*
      float start_x = ((float) ((index % (model_x*model_y)) % model_x));
      float start_y = ((float) ((index / model_x) % model_y));
      float start_z = ((float) (index / (model_x*model_y)));
    */
    float new_x, new_y, new_z;
    /* This is just a matrix multiplication with rotation */
    new_x = model_x/2. - 0.5 + (rotation_matrix_00*start_x +
				rotation_matrix_01*start_y +
				rotation_matrix_02*start_z);
    new_y = model_y/2. - 0.5 + (rotation_matrix_10*start_x +
				rotation_matrix_11*start_y +
				rotation_matrix_12*start_z);
    new_z = model_z/2. - 0.5 + (rotation_matrix_20*start_x +
				rotation_matrix_21*start_y +
				rotation_matrix_22*start_z);
    rotated_model[index] = device_model_get(model,
					    model_x, model_y, model_z,
					    new_x, new_y, new_z);
  }
}
