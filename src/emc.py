import emc_cuda
import afnumpy

_MAX_PHOTON_COUNT = 100

def set_to_value(array, value):
    size = len(array.flatten())
    array_pointer = emc_cuda.int_to_float_pointer(array.d_array.device_ptr())
    emc_cuda.set_to_value(array_pointer, size, value)

def masked_set(array, mask, value):
    if array.shape != mask.shape:
        raise ValueError("Array and mask must have the same shape")
    #array = afnumpy.float32(array)
    if  array.dtype != afnumpy.float32:
        raise ValueError("Array must be of type float32")
    #mask = afnumpy.int32(mask)
    size = len(array.flatten())
    array_pointer = emc_cuda.int_to_float_pointer(array.d_array.device_ptr())
    mask_pointer = emc_cuda.int_to_int_pointer(mask.d_array.device_ptr())
    emc_cuda.masked_set(array_pointer, mask_pointer, size, value)

def equivalent_sigma(number_of_pixels):
    return 1./numpy.sqrt(number_of_pixels)

def expand_model(model, slices, rotations, coordinates):
    if len(slices) != len(rotations):
        raise ValueError("Slices and rotations must be of the same length.")
    if len(model.shape) != 3:
        raise ValueError("Model must be a 3D array.")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 2 or rotations.shape[1] != 4:
        raise ValueError("rotations must be a nx4 array.")
    if len(coordinates.shape) != 3 or coordinates.shape[0] != 3 or coordinates.shape[1:] != slices.shape[1:]:
        raise ValueError("coordinates must be 3xXxY array where X and Y are the dimensions of the slices.")

    number_of_rotations = len(rotations)
    model_pointer = emc_cuda.int_to_float_pointer(model.d_array.device_ptr())
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    rotations_pointer = emc_cuda.int_to_float_pointer(rotations.d_array.device_ptr())
    coordinates_pointer = emc_cuda.int_to_float_pointer(coordinates.d_array.device_ptr())
    emc_cuda.cuda_expand_model(model_pointer, model.shape[2], model.shape[1], model.shape[0],
                               slices_pointer, slices.shape[2], slices.shape[1],
                               rotations_pointer, number_of_rotations,
                               coordinates_pointer)

def get_slice(model, rotation, coordinates):
    if len(rotation) != 4:
        raise ValueError("rotations must be a quaternion (len 4 array)")
    if len(model.shape) != 3:
        raise ValueError("Model must be a 3D array.")
    if len(coordinates.shape) != 3 or coordinates.shape[0] != 3:
        raise ValueError("coordinates must be 3xXxY.")

    model = afnumpy.array(model, dtype="float32")
    this_slice = afnumpy.zeros(coordinates.shape[1:], dtype="float32").reshape((1,) + coordinates.shape[1:])
    rotation = afnumpy.array(rotation, dtype="float32").reshape((1, 4))
    model_pointer = emc_cuda.int_to_float_pointer(model.d_array.device_ptr())
    this_slice_pointer = emc_cuda.int_to_float_pointer(this_slice.d_array.device_ptr())
    rotation_pointer = emc_cuda.int_to_float_pointer(rotation.d_array.device_ptr())
    coordinates_pointer = emc_cuda.int_to_float_pointer(coordinates.d_array.device_ptr())
    emc_cuda.cuda_expand_model(model_pointer, model.shape[2], model.shape[1], model.shape[0],
                               this_slice_pointer, coordinates.shape[2], coordinates.shape[1],
                               rotation_pointer, 1,
                               coordinates_pointer)
    return this_slice
    

def insert_slices(model, model_weights, slices, slice_weights, rotations, coordinates):
    if len(slices) != len(rotations):
        raise ValueError("slices and rotations must be of the same length.")
    if len(slices) != len(slice_weights):
        raise ValueError("slices and slice_weights must be of the same length.")
    if len(slice_weights.shape) != 1:
        raise ValueError("slice_weights must be one dimensional.")
    if len(model.shape) != 3 or model.shape != model_weights.shape:
        raise ValueError("model and model_weights must be 3D arrays of the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 2 or rotations.shape[1] != 4:
        raise ValueError("rotations must be a nx4 array.")
    if len(coordinates.shape) != 3 or coordinates.shape[0] != 3 or coordinates.shape[1:] != slices.shape[1:]:
        raise ValueError("coordinates must be 3xXxY array where X and Y are the dimensions of the slices.")

    number_of_rotations = len(rotations)
    model_pointer = emc_cuda.int_to_float_pointer(model.d_array.device_ptr())
    model_weights_pointer = emc_cuda.int_to_float_pointer(model_weights.d_array.device_ptr())
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    slice_weights_pointer = emc_cuda.int_to_float_pointer(slice_weights.d_array.device_ptr())
    rotations_pointer = emc_cuda.int_to_float_pointer(rotations.d_array.device_ptr())
    coordinates_pointer = emc_cuda.int_to_float_pointer(coordinates.d_array.device_ptr())
    emc_cuda.cuda_insert_slices(model_pointer, model_weights_pointer,
                                model.shape[2], model.shape[1], model.shape[0],
                                slices_pointer, slices.shape[2], slices.shape[1],
                                slice_weights_pointer,
                                rotations_pointer, number_of_rotations,
                                coordinates_pointer)


def update_slices(slices, patterns, responsabilities):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array.")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    patterns_pointer = emc_cuda.int_to_float_pointer(patterns.d_array.device_ptr())
    responsabilities_pointer = emc_cuda.int_to_float_pointer(responsabilities.d_array.device_ptr())
    emc_cuda.cuda_update_slices(slices_pointer, slices.shape[0], patterns_pointer, patterns.shape[0], patterns.shape[2], patterns.shape[1], responsabilities_pointer)
    
    
def calculate_responsabilities(patterns, slices, responsabilities, sigma):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    #sigma = afnumpy.float32(sigma)
    if sigma <= 0.: raise ValueError("sigma must be larger than zeros")
    patterns_pointer = emc_cuda.int_to_float_pointer(patterns.d_array.device_ptr())
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    responsabilities_pointer = emc_cuda.int_to_float_pointer(responsabilities.d_array.device_ptr())
    emc_cuda.cuda_calculate_responsabilities(patterns_pointer, patterns.shape[0], slices_pointer, slices.shape[0],
                                             slices.shape[2], slices.shape[1], responsabilities_pointer, sigma)

"""void cuda_calculate_responsabilities_poisson(const float *const patterns, const int number_of_patterns,
					     const float *const slices, const int number_of_rotations,
					     const int image_x, const int image_y,
					     float *const responsabilities, const float *constlog_factorial_table);
"""

def _log_factorial_table(max_value):
    if max_value > _MAX_PHOTON_COUNT:
        raise ValueError("Poisson values can not be used with photon counts higher than {0}".format(_MAX_PHOTON_COUNT))
    log_factorial_table = afnumpy.zeros(max_value+1, dtype="float32")
    log_factorial_table[0] = 0.
    for i in range(1, int(max_value+1)):
        log_factorial_table[i] = log_factorial_table[i-1] + afnumpy.log(i)
    return log_factorial_table


def calculate_responsabilities_poisson(patterns, slices, responsabilities):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if (calculate_responsabilities_poisson.log_factorial_table is None or
        len(calculate_responsabilities_poisson.log_factorial_table) <= patterns.max()):
        calculate_responsabilities_poisson.log_factorial_table = _log_factorial_table(patterns.max())

    patterns_pointer = emc_cuda.int_to_float_pointer(patterns.d_array.device_ptr())
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    responsabilities_pointer = emc_cuda.int_to_float_pointer(responsabilities.d_array.device_ptr())
    log_factorial_table_pointer = emc_cuda.int_to_float_pointer(calculate_responsabilities_poisson.log_factorial_table.d_array.device_ptr())
    emc_cuda.cuda_calculate_responsabilities_poisson(patterns_pointer, patterns.shape[0], slices_pointer, slices.shape[0],
                                                     slices.shape[2], slices.shape[1], responsabilities_pointer, log_factorial_table_pointer)
calculate_responsabilities_poisson.log_factorial_table = None
    

def calculate_responsabilities_sparse(patterns, slices, responsabilities):
    if not isinstance(patterns, dict):
        raise ValueError("patterns must be a dictionary containing the keys: indcies, values and start_indices")
    if ("indices" not in patterns or
        "values" not in patterns or
        "start_indices" not in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
    if len(responsabilities.shape) != 2: raise ValueError("responsabilities must have shape nrotations x npatterns")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    
    if (calculate_responsabilities_sparse.log_factorial_table is None or
        len(calculate_responsabilities_sparse.log_factorial_table) <= patterns["values"].max()):
        calculate_responsabilities_sparse.log_factorial_table = _log_factorial_table(patterns["values"].max())
    
    if (calculate_responsabilities_sparse.slice_sums is None or
        len(calculate_responsabilities_sparse.slice_sums) != len(slices)):
        calculate_responsabilities_sparse.slice_sums = afnumpy.empty(len(slices), dtype=afnumpy.float32)

    patterns_indices_pointer = emc_cuda.int_to_int_pointer(patterns["indices"].d_array.device_ptr())
    patterns_values_pointer = emc_cuda.int_to_float_pointer(patterns["values"].d_array.device_ptr())
    patterns_start_indices_pointer = emc_cuda.int_to_int_pointer(patterns["start_indices"].d_array.device_ptr())
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    responsabilities_pointer = emc_cuda.int_to_float_pointer(responsabilities.d_array.device_ptr())
    slice_sums_pointer = emc_cuda.int_to_float_pointer(calculate_responsabilities_sparse.slice_sums.d_array.device_ptr())
    log_factorial_table_pointer = emc_cuda.int_to_float_pointer(calculate_responsabilities_sparse.log_factorial_table.d_array.device_ptr())
    number_of_patterns = len(patterns["start_indices"])-1
    number_of_rotations = len(slices)
    emc_cuda.cuda_calculate_responsabilities_sparse(patterns_start_indices_pointer,
                                                    patterns_indices_pointer,
                                                    patterns_values_pointer,
                                                    number_of_patterns,
                                                    slices_pointer,
                                                    number_of_rotations,
                                                    slices.shape[2], slices.shape[1],
                                                    responsabilities_pointer,
                                                    slice_sums_pointer,
                                                    log_factorial_table_pointer)
calculate_responsabilities_sparse.log_factorial_table = None
calculate_responsabilities_sparse.slice_sums = None

"""
void cuda_update_slices_sparse(float *const slices, const int number_of_rotations, const int *const pattern_start_indices,
			       const int *const pattern_indices, const float *const pattern_values, const int number_of_patterns,
			       const int image_x, const int image_y, const float *const responsabilities));
"""
def update_slices_sparse(slices, patterns, responsabilities):
    if (not "indices" in patterns or
        not "values" in patterns or
        not "start_indices" in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
    if len(responsabilities.shape) != 2: raise ValueError("responsabilities must have shape nrotations x npatterns")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    patterns_indices_pointer = emc_cuda.int_to_float_pointer(patterns["indices"].d_array.device_ptr())
    patterns_values_pointer = emc_cuda.int_to_float_pointer(patterns["values"].d_array.device_ptr())
    patterns_start_indices_pointer = emc_cuda.int_to_float_pointer(patterns["start_indices"].d_array.device_ptr())
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    responsabilities_pointer = emc_cuda.int_to_float_pointer(responsabilities.d_array.device_ptr())
    slice_sums_pointer = emc_cuda.int_to_float_pointer(slice_sums.d_array.device_ptr())
    log_factorial_table_pointer = emc_cuda.int_to_float_pointer(calculate_responsabilities_sparse.slice_sums.d_array.device_ptr())
    number_of_patterns = len(patterns["start_indices"])-1
    number_of_rotations = len(slices)
    emc_cuda.cuda_update_slices_sparse(slices_pointer, number_of_rotations, pattern_start_indices_pointer,
                                       pattern_indices_pointer, pattern_values_pointer, number_of_patterns,
                                       slices.shape[2], slices.shape[1], responsabilities_pointer)

    
def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size):
    pixels_to_im = pixel_size/detector_distance/wavelength
    x0_pixels = afnumpy.arange(image_shape[0], dtype="float32") - image_shape[0]/2 + 0.5
    x1_pixels = afnumpy.arange(image_shape[1], dtype="float32") - image_shape[1]/2 + 0.5
    x0 = x0_pixels*pixels_to_im
    x1 = x1_pixels*pixels_to_im
    r_pixels = afnumpy.sqrt(x0_pixels[:, afnumpy.newaxis]**2 + x1_pixels[afnumpy.newaxis, :]**2)
    theta = afnumpy.arctan(r_pixels*pixel_size / detector_distance)
    x2 = 1./wavelength*(1 - afnumpy.cos(theta))
    x2_pixels = x2/pixels_to_im

    x0_2d, x1_2d = afnumpy.meshgrid(x0_pixels, x1_pixels, indexing="ij")
    output_coordinates = afnumpy.zeros((3, ) + image_shape, dtype="float32")
    #return output_coordinates, x0_2d
    # print "output_coordinates.shape = {0}".format(str(type(output_coordinates)))
    # print "x0_2d.shape = {0}".format(str(type(x0_2d)))
    output_coordinates[0, :, :] = x0_2d
    output_coordinates[1, :, :] = x1_2d
    output_coordinates[2, :, :] = x2_pixels
    return output_coordinates


def rotate_model(model, rotated_model, rotation):
    rotation = afnumpy.array(rotation, dtype="float32")
    if model.shape != rotated_model.shape:
        raise ValueError("model and rotated_model must have the same shape")
    if len(rotation.shape) != 1 or rotation.shape[0] != 4:
        raise ValueError("rotation must be length 4 arrray")
    model_pointer = emc_cuda.int_to_float_pointer(model.d_array.device_ptr())
    rotated_model_pointer = emc_cuda.int_to_float_pointer(rotated_model.d_array.device_ptr())
    rotation_pointer = emc_cuda.int_to_float_pointer(rotation.d_array.device_ptr())
    emc_cuda.cuda_rotate_model(model_pointer, rotated_model_pointer, model.shape[2], model.shape[1], model.shape[0], rotation_pointer)
