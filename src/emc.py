import emc_cuda
import afnumpy

_MAX_PHOTON_COUNT = 1500
_INTERPOLATION = {"nearest_neighbour": 0,
                  "linear": 1}

def _get_pointer(array):
    if array.dtype == afnumpy.float32:
        return emc_cuda.int_to_float_pointer(array.d_array.device_ptr() + 4*array.d_array.offset())
    elif array.dtype == afnumpy.int32:
        return emc_cuda.int_to_int_pointer(array.d_array.device_ptr() + 4*array.d_array.offset())
    else:
        raise ValueError("_get_pointer received argument with unrecognized dtype: {0}".format(array.dtype))

def set_to_value(array, value):
    size = len(array.flatten())
    emc_cuda.set_to_value(_get_pointer(array), size, value)

def masked_set(array, mask, value):
    if array.shape != mask.shape:
        raise ValueError("Array and mask must have the same shape")
    #array = afnumpy.float32(array)
    if  array.dtype != afnumpy.float32:
        raise ValueError("Array must be of type float32")
    #mask = afnumpy.int32(mask)
    size = len(array.flatten())
    emc_cuda.masked_set(_get_pointer(array),
                        _get_pointer(mask),
                        size, value)

def equivalent_sigma(number_of_pixels):
    return 1./afnumpy.sqrt(number_of_pixels)

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
    emc_cuda.cuda_expand_model(_get_pointer(model),
                               model.shape[2], model.shape[1], model.shape[0],
                               _get_pointer(slices),
                               slices.shape[2], slices.shape[1],
                               _get_pointer(rotations),
                               number_of_rotations,
                               _get_pointer(coordinates))
    
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
    emc_cuda.cuda_expand_model(_get_pointer(model),
                               model.shape[2], model.shape[1], model.shape[0],
                               _get_pointer(this_slice),
                               coordinates.shape[2], coordinates.shape[1],
                               _get_pointer(rotations), 1,
                               _get_pointer(coordinates))
    return this_slice
    
def insert_slices(model, model_weights, slices, slice_weights, rotations, coordinates, interpolation="linear"):
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

    interpolation_int = _INTERPOLATION[interpolation]
    number_of_rotations = len(rotations)
    emc_cuda.cuda_insert_slices(_get_pointer(model),
                                _get_pointer(model_weights),
                                model.shape[2], model.shape[1], model.shape[0],
                                _get_pointer(slices),
                                slices.shape[2], slices.shape[1],
                                _get_pointer(slice_weights),
                                _get_pointer(rotations),
                                number_of_rotations,
                                _get_pointer(coordinates),
                                interpolation_int)

def insert_slices_partial(partial_model, partial_model_weights, full_model_shape, partial_model_corner,
                          slices, slice_weights, rotations, coordinates, interpolation="linear"):
    if len(slices) != len(rotations):
        raise ValueError("slices and rotations must be of the same length.")
    if len(slices) != len(slice_weights):
        raise ValueError("slices and slice_weights must be of the same length.")
    if len(slice_weights.shape) != 1:
        raise ValueError("slice_weights must be one dimensional.")
    if len(partial_model.shape) != 3 or partial_model.shape != partial_model_weights.shape:
        raise ValueError("partial_model and partial_model_weights must be 3D arrays of the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 2 or rotations.shape[1] != 4:
        raise ValueError("rotations must be a nx4 array.")
    if len(coordinates.shape) != 3 or coordinates.shape[0] != 3 or coordinates.shape[1:] != slices.shape[1:]:
        raise ValueError("coordinates must be 3xXxY array where X and Y are the dimensions of the slices.")

    interpolation_int = _INTERPOLATION[interpolation]
    number_of_rotations = len(rotations)
    emc_cuda.cuda_insert_slices_partial(_get_pointer(partial_model),
                                        _get_pointer(partial_model_weights),
                                        full_model_shape[2], partial_model_corner[2],
                                        partial_model_corner[2]+partial_model.shape[2],
                                        full_model_shape[1], partial_model_corner[1],
                                        partial_model_corner[1]+partial_model.shape[1],
                                        full_model_shape[0], partial_model_corner[0],
                                        partial_model_corner[0]+partial_model.shape[0],
                                        _get_pointer(slices),
                                        slices.shape[2], slices.shape[1],
                                        _get_pointer(slice_weights),
                                        _get_pointer(rotations),
                                        number_of_rotations,
                                        _get_pointer(coordinates),
                                        interpolation_int)
    
def update_slices(slices, patterns, responsabilities, scalings=None):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array.")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if scalings is not None and scalings.shape != responsabilities.shape:
        raise ValueError("Scalings must have the same shape as responsabilities")

    if scalings is None:
        emc_cuda.cuda_update_slices(_get_pointer(slices),
                                    slices.shape[0],
                                    _get_pointer(patterns),
                                    patterns.shape[0], patterns.shape[2], patterns.shape[1],
                                    _get_pointer(responsabilities))
    else:
        emc_cuda.cuda_update_slices_scaling(_get_pointer(slices),
                                            slices.shape[0],
                                            _get_pointer(patterns),
                                            patterns.shape[0], patterns.shape[2], patterns.shape[1],
                                            _get_pointer(responsabilities),
                                            _get_pointer(scalings))
        
    
def calculate_responsabilities(patterns, slices, responsabilities, sigma):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    #sigma = afnumpy.float32(sigma)
    if sigma <= 0.: raise ValueError("sigma must be larger than zeros")
    emc_cuda.cuda_calculate_responsabilities(_get_pointer(patterns),
                                             patterns.shape[0],
                                             _get_pointer(slices),
                                             slices.shape[0], slices.shape[2], slices.shape[1],
                                             _get_pointer(responsabilities),
                                             sigma)

def _log_factorial_table(max_value):
    if max_value > _MAX_PHOTON_COUNT:
        raise ValueError("Poisson values can not be used with photon counts higher than {0}".format(_MAX_PHOTON_COUNT))
    log_factorial_table = afnumpy.zeros(int(max_value+1), dtype="float32")
    log_factorial_table[0] = 0.
    for i in range(1, int(max_value+1)):
        log_factorial_table[i] = log_factorial_table[i-1] + afnumpy.log(i)
    return log_factorial_table

def calculate_responsabilities_poisson(patterns, slices, responsabilities, scalings=None):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if (calculate_responsabilities_poisson.log_factorial_table is None or
        len(calculate_responsabilities_poisson.log_factorial_table) <= patterns.max()):
        calculate_responsabilities_poisson.log_factorial_table = _log_factorial_table(patterns.max())
    if scalings is not None and scalings.shape != responsabilities.shape:
        raise ValueError("Scalings must have the same shape as responsabilities")
    if scalings is None:
        emc_cuda.cuda_calculate_responsabilities_poisson(_get_pointer(patterns),
                                                         patterns.shape[0],
                                                         _get_pointer(slices),
                                                         slices.shape[0], slices.shape[2], slices.shape[1],
                                                         _get_pointer(responsabilities),
                                                         _get_pointer(calculate_responsabilities_poisson.log_factorial_table))
    else:
        emc_cuda.cuda_calculate_responsabilities_poisson_scaling(_get_pointer(patterns),
                                                                 patterns.shape[0],
                                                                 _get_pointer(slices),
                                                                 slices.shape[0], slices.shape[2], slices.shape[1],
                                                                 _get_pointer(scalings),
                                                                 _get_pointer(responsabilities),
                                                                 _get_pointer(calculate_responsabilities_poisson.log_factorial_table))
calculate_responsabilities_poisson.log_factorial_table = None
        
def calculate_responsabilities_sparse(patterns, slices, responsabilities, scalings=None):
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
    if scalings is not None and scalings.shape != responsabilities.shape:
        raise ValueError("Scalings must have the same shape as responsabilities")
    
    if (calculate_responsabilities_sparse.log_factorial_table is None or
        len(calculate_responsabilities_sparse.log_factorial_table) <= patterns["values"].max()):
        calculate_responsabilities_sparse.log_factorial_table = _log_factorial_table(patterns["values"].max())
    
    if (calculate_responsabilities_sparse.slice_sums is None or
        len(calculate_responsabilities_sparse.slice_sums) != len(slices)):
        calculate_responsabilities_sparse.slice_sums = afnumpy.empty(len(slices), dtype=afnumpy.float32)

    number_of_patterns = len(patterns["start_indices"])-1
    number_of_rotations = len(slices)
    if scalings is None:
        emc_cuda.cuda_calculate_responsabilities_sparse(_get_pointer(patterns["start_indices"]),
                                                        _get_pointer(patterns["indices"]),
                                                        _get_pointer(patterns["values"]),
                                                        number_of_patterns,
                                                        _get_pointer(slices),
                                                        number_of_rotations,
                                                        slices.shape[2], slices.shape[1],
                                                        _get_pointer(responsabilities),
                                                        _get_pointer(calculate_responsabilities_sparse.slice_sums),
                                                        _get_pointer(calculate_responsabilities_sparse.log_factorial_table))
    else:
        emc_cuda.cuda_calculate_responsabilities_sparse_scaling(_get_pointer(patterns["start_indices"]),
                                                                _get_pointer(patterns["indices"]),
                                                                _get_pointer(patterns["values"]),
                                                                number_of_patterns,
                                                                _get_pointer(slices),
                                                                number_of_rotations,
                                                                slices.shape[2], slices.shape[1],
                                                                _get_pointer(scalings),
                                                                _get_pointer(responsabilities),
                                                                _get_pointer(calculate_responsabilities_sparse.slice_sums),
                                                                _get_pointer(calculate_responsabilities_sparse.log_factorial_table))

calculate_responsabilities_sparse.log_factorial_table = None
calculate_responsabilities_sparse.slice_sums = None
def update_slices_sparse(slices, patterns, responsabilities, scalings=None):
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
    if scalings is not None and scalings.shape != responsabilities.shape:
        raise ValueError("Scalings must have the same shape as responsabilities")
    number_of_patterns = len(patterns["start_indices"])-1
    number_of_rotations = len(slices)
    if scalings is None:
        emc_cuda.cuda_update_slices_sparse(_get_pointer(slices),
                                           number_of_rotations,
                                           _get_pointer(patterns["start_indices"]),
                                           _get_pointer(patterns["indices"]),
                                           _get_pointer(patterns["values"]),
                                           number_of_patterns,
                                           slices.shape[2],
                                           slices.shape[1],
                                           _get_pointer(responsabilities))
    else:
        emc_cuda.cuda_update_slices_sparse_scaling(_get_pointer(slices),
                                                   number_of_rotations,
                                                   _get_pointer(patterns["start_indices"]),
                                                   _get_pointer(patterns["indices"]),
                                                   _get_pointer(patterns["values"]),
                                                   number_of_patterns,
                                                   slices.shape[2],
                                                   slices.shape[1],
                                                   _get_pointer(responsabilities),
                                                   _get_pointer(scalings))

def calculate_scaling_poisson(patterns, slices, scaling):
    if len(patterns.shape) != 3:
        raise ValueError("Patterns must be a 3D array")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 2:
        raise ValueError("Slices must be a 2D array")
    if slices.shape[1:] != patterns.shape[1:]:
        raise ValueError("Slices and patterns must be the same shape")
    if scaling.shape[0] != slices.shape[0] or scaling.shape[1] != patterns.shape[0]:
        raise ValueError("scaling must have shape nrotations x npatterns")        
    emc_cuda.cuda_calculate_scaling_poisson(_get_pointer(patterns),
                                            patterns.shape[0],
                                            _get_pointer(slices),
                                            slices.shape[0],
                                            afnumpy.prod(slices.shape[1:]),
                                            _get_pointer(scaling))

def calculate_scaling_poisson_sparse(patterns, slices, scaling):
    if not isinstance(patterns, dict):
        raise ValueError("patterns must be a dictionary containing the keys: indcies, values and start_indices")
    if ("indices" not in patterns or
        "values" not in patterns or
        "start_indices" not in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != scaling.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 2:
        raise ValueError("Slices must be a 2D array")
    number_of_patterns = len(patterns["start_indices"])-1
    if scaling.shape[0] != slices.shape[0] or scaling.shape[1] != number_of_patterns:
        raise ValueError("scaling must have shape nrotations x npatterns")        
    emc_cuda.cuda_calculate_scaling_poisson_sparse(_get_pointer(patterns["start_indices"]),
                                                   _get_pointer(patterns["indices"]),
                                                   _get_pointer(patterns["values"]),
                                                   number_of_patterns,
                                                   _get_pointer(slices),
                                                   slices.shape[0],
                                                   afnumpy.prod(slices.shape[1:]),
                                                   _get_pointer(scaling))


# Attempt to fix for high angles
def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size):
    x_pixels_1d = afnumpy.arange(image_shape[1]) - image_shape[1]/2. + 0.5
    y_pixels_1d = afnumpy.arange(image_shape[0]) - image_shape[0]/2. + 0.5
    y_pixels, x_pixels = afnumpy.meshgrid(y_pixels_1d, x_pixels_1d, indexing="ij")
    x_meters = x_pixels*pixel_size
    y_meters = y_pixels*pixel_size
    radius_meters = afnumpy.sqrt(x_meters**2 + y_meters**2)

    scattering_angle = afnumpy.arctan(radius_meters / detector_distance)
    z = -1./wavelength*(1. - afnumpy.cos(scattering_angle))
    radius_fourier = afnumpy.sqrt(1./wavelength**2 - (1./wavelength - abs(z))**2)

    x = x_meters * radius_fourier / radius_meters
    y = y_meters * radius_fourier / radius_meters

    output_coordinates = afnumpy.zeros((3, ) + image_shape, dtype=afnumpy.float32)
    output_coordinates[0, :, :] = afnumpy.float32(x)
    output_coordinates[1, :, :] = afnumpy.float32(y)
    output_coordinates[2, :, :] = afnumpy.float32(z)

    # Rescale so that edge pixels match.
    furthest_edge_coordinate = afnumpy.sqrt(x[0, image_shape[1]/2]**2 + y[0, image_shape[1]/2]**2 + z[0, image_shape[1]/2]**2)
    rescale_factor = image_shape[0]/2./furthest_edge_coordinate
    output_coordinates *= rescale_factor
    
    return output_coordinates

# Old standard
def ewald_coordinates_old(image_shape, wavelength, detector_distance, pixel_size):
    pixels_to_im = pixel_size/detector_distance/wavelength
    x_pixels = afnumpy.arange(image_shape[0], dtype="float32") - image_shape[1]/2 + 0.5
    y_pixels = afnumpy.arange(image_shape[1], dtype="float32") - image_shape[0]/2 + 0.5
    x = x_pixels*pixels_to_im
    y = y_pixels*pixels_to_im
    r_pixels = afnumpy.sqrt(x_pixels[afnumpy.newaxis, :]**2 + y_pixels[:, afnumpy.newaxis]**2)
    theta = afnumpy.arctan(r_pixels*pixel_size / detector_distance)
    z = -1./wavelength*(1 - afnumpy.cos(theta))
    z_pixels = z/pixels_to_im

    y_2d, x_2d = afnumpy.meshgrid(y_pixels, x_pixels, indexing="ij")
    #x_2d, y_2d = afnumpy.meshgrid(x_pixels, y_pixels, indexing="ij")
    output_coordinates = afnumpy.zeros((3, ) + image_shape, dtype="float32")
    output_coordinates[0, :, :] = x_2d
    output_coordinates[1, :, :] = y_2d
    output_coordinates[2, :, :] = z_pixels
    return output_coordinates

def rotate_model(model, rotated_model, rotation):
    rotation = afnumpy.array(rotation, dtype="float32")
    if model.shape != rotated_model.shape:
        raise ValueError("model and rotated_model must have the same shape")
    if len(rotation.shape) != 1 or rotation.shape[0] != 4:
        raise ValueError("rotation must be length 4 arrray")
    emc_cuda.cuda_rotate_model(_get_pointer(model),
                               _get_pointer(rotated_model),
                               model.shape[2], model.shape[1], model.shape[0],
                               _get_pointer(rotation))
                               

