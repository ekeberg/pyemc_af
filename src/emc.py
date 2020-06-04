try:
    import emc_cuda
except ImportError:
    emc_cuda = None
try:
    import emc_cpu
except ImportError:
    emc_cpu = None
import afnumpy
import numpy

if emc_cuda is None and emc_cpu is None:
    raise ImportError("Could not load any emc backend (cuda or cpu)")
    
default_backend = [emc_cuda if emc_cuda else emc_cpu]

_MAX_PHOTON_COUNT = 1500
_INTERPOLATION = {"nearest_neighbour": 0,
                  "linear": 1}

def set_backend(backend):
    if backend == "cuda":
        if emc_cuda is None:
            raise ImportError("CUDA backend is not available")
        default_backend[0] = emc_cuda
        afnumpy.arrayfire.backend.set_unsafe("cuda")
    elif backend == "cpu":
        if emc_cpu is None:
            raise ImportError("CPU backend is not available")
        default_backend[0] = emc_cpu
        afnumpy.arrayfire.backend.set_unsafe("cpu")
    else:
        raise ValueError("No backend called {}".format(backend))

def _get_pointer(array, backend=default_backend):
    if array.dtype == afnumpy.float32:
        return backend[0].int_to_float_pointer(array.d_array.device_ptr() + 4*array.d_array.offset())
    elif array.dtype == afnumpy.int32:
        return backend[0].int_to_int_pointer(array.d_array.device_ptr() + 4*array.d_array.offset())
    else:
        raise ValueError("_get_pointer received argument with unrecognized dtype: {0}".format(array.dtype))

def set_to_value(array, value, backend=default_backend):
    size = len(array.flatten())
    backend[0].set_to_value(_get_pointer(array), size, value)

def masked_set(array, mask, value, backend=default_backend):
    if array.shape != mask.shape:
        raise ValueError("Array and mask must have the same shape")
    #array = afnumpy.float32(array)
    if  array.dtype != afnumpy.float32:
        raise ValueError("Array must be of type float32")
    #mask = afnumpy.int32(mask)
    size = len(array.flatten())
    backend[0].masked_set(_get_pointer(array),
                          _get_pointer(mask),
                          size, value)

def equivalent_sigma(number_of_pixels):
    return 1./afnumpy.sqrt(number_of_pixels)

def expand_model(model, slices, rotations, coordinates, backend=default_backend):
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
    backend[0].expand_model(_get_pointer(model),
                            model.shape[2], model.shape[1], model.shape[0],
                            _get_pointer(slices),
                            slices.shape[2], slices.shape[1],
                            _get_pointer(rotations),
                            number_of_rotations,
                            _get_pointer(coordinates))
    
def get_slice(model, rotation, coordinates, backend=default_backend):
    if len(rotation) != 4:
        raise ValueError("rotations must be a quaternion (len 4 array)")
    if len(model.shape) != 3:
        raise ValueError("Model must be a 3D array.")
    if len(coordinates.shape) != 3 or coordinates.shape[0] != 3:
        raise ValueError("coordinates must be 3xXxY.")

    model = afnumpy.array(model, dtype="float32")
    this_slice = afnumpy.zeros(coordinates.shape[1:], dtype="float32").reshape((1,) + coordinates.shape[1:])
    rotation = afnumpy.array(rotation, dtype="float32").reshape((1, 4))
    backend[0].expand_model(_get_pointer(model),
                            model.shape[2], model.shape[1], model.shape[0],
                            _get_pointer(this_slice),
                            coordinates.shape[2], coordinates.shape[1],
                            _get_pointer(rotations), 1,
                            _get_pointer(coordinates))
    return this_slice
    
def insert_slices(model, model_weights, slices, slice_weights, rotations, coordinates, interpolation="linear",
                  backend=default_backend):
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
    backend[0].insert_slices(_get_pointer(model),
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
                          slices, slice_weights, rotations, coordinates, interpolation="linear",
                          backend=default_backend):
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
    backend[0].insert_slices_partial(_get_pointer(partial_model),
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
    
def update_slices(slices, patterns, responsabilities, scalings=None,
                  backend=default_backend):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array.")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 and scalings.shape[0] == patterns.shape[0])):
        raise ValueError("Scalings must have the same shape as responsabilities")

    if scalings is None:
        backend[0].update_slices(_get_pointer(slices),
                                 slices.shape[0],
                                 _get_pointer(patterns),
                                 patterns.shape[0], patterns.shape[2], patterns.shape[1],
                                 _get_pointer(responsabilities))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        backend[0].update_slices_scaling(_get_pointer(slices),
                                         slices.shape[0],
                                         _get_pointer(patterns),
                                         patterns.shape[0], patterns.shape[2], patterns.shape[1],
                                         _get_pointer(responsabilities),
                                         _get_pointer(scalings))
    else:
        # Scaling per pattern
        backend[0].update_slices_per_pattern_scaling(_get_pointer(slices),
                                                     slices.shape[0],
                                                     _get_pointer(patterns),
                                                     patterns.shape[0], patterns.shape[2], patterns.shape[1],
                                                     _get_pointer(responsabilities),
                                                     _get_pointer(scalings))
        
def _log_factorial_table(max_value):
    if max_value > _MAX_PHOTON_COUNT:
        raise ValueError("Poisson values can not be used with photon counts higher than {0}".format(_MAX_PHOTON_COUNT))
    log_factorial_table = afnumpy.zeros(int(max_value+1), dtype="float32")
    log_factorial_table[0] = 0.
    for i in range(1, int(max_value+1)):
        log_factorial_table[i] = log_factorial_table[i-1] + afnumpy.log(i)
    return log_factorial_table

def calculate_responsabilities_poisson(patterns, slices, responsabilities, scalings=None,
                                       backend=default_backend):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if (calculate_responsabilities_poisson.log_factorial_table is None or
        len(calculate_responsabilities_poisson.log_factorial_table) <= patterns.max()):
        calculate_responsabilities_poisson.log_factorial_table = _log_factorial_table(patterns.max())
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 or scalings.shape[0] == patterns.shape[0])):
        raise ValueError("Scalings must have the same shape as responsabilities")
    if scalings is None:
        backend[0].calculate_responsabilities_poisson(_get_pointer(patterns),
                                                      patterns.shape[0],
                                                      _get_pointer(slices),
                                                      slices.shape[0], slices.shape[2], slices.shape[1],
                                                      _get_pointer(responsabilities),
                                                      _get_pointer(calculate_responsabilities_poisson.log_factorial_table))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        backend[0].calculate_responsabilities_poisson_scaling(_get_pointer(patterns),
                                                              patterns.shape[0],
                                                              _get_pointer(slices),
                                                              slices.shape[0], slices.shape[2], slices.shape[1],
                                                              _get_pointer(scalings),
                                                              _get_pointer(responsabilities),
                                                              _get_pointer(calculate_responsabilities_poisson.log_factorial_table))
    else:
        # Scaling per pattern
        backend[0].calculate_responsabilities_poisson_per_pattern_scaling(_get_pointer(patterns),
                                                                          patterns.shape[0],
                                                                          _get_pointer(slices),
                                                                          slices.shape[0], slices.shape[2], slices.shape[1],
                                                                          _get_pointer(scalings),
                                                                          _get_pointer(responsabilities),
                                                                          _get_pointer(calculate_responsabilities_poisson.log_factorial_table))
calculate_responsabilities_poisson.log_factorial_table = None
        
def calculate_responsabilities_sparse(patterns, slices, responsabilities, scalings=None,
                                      backend=default_backend):
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
    number_of_patterns = len(patterns["start_indices"])-1
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 or scalings.shape[0] == number_of_patterns)):
        raise ValueError("Scalings must have the same shape as responsabilities")
    
    if (calculate_responsabilities_sparse.log_factorial_table is None or
        len(calculate_responsabilities_sparse.log_factorial_table) <= patterns["values"].max()):
        calculate_responsabilities_sparse.log_factorial_table = _log_factorial_table(patterns["values"].max())
    
    if (calculate_responsabilities_sparse.slice_sums is None or
        len(calculate_responsabilities_sparse.slice_sums) != len(slices)):
        calculate_responsabilities_sparse.slice_sums = afnumpy.empty(len(slices), dtype=afnumpy.float32)

    number_of_rotations = len(slices)
    if scalings is None:
        backend[0].calculate_responsabilities_sparse(_get_pointer(patterns["start_indices"]),
                                                     _get_pointer(patterns["indices"]),
                                                     _get_pointer(patterns["values"]),
                                                     number_of_patterns,
                                                     _get_pointer(slices),
                                                     number_of_rotations,
                                                     slices.shape[2], slices.shape[1],
                                                     _get_pointer(responsabilities),
                                                     _get_pointer(calculate_responsabilities_sparse.slice_sums),
                                                     _get_pointer(calculate_responsabilities_sparse.log_factorial_table))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        backend[0].calculate_responsabilities_sparse_scaling(_get_pointer(patterns["start_indices"]),
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
    else:
        # Scaling per pattern
        backend[0].calculate_responsabilities_sparse_per_pattern_scaling(_get_pointer(patterns["start_indices"]),
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
def update_slices_sparse(slices, patterns, responsabilities, scalings=None, resp_threshold=0.,
                         backend=default_backend):
    if (not "indices" in patterns or
        not "values" in patterns or
        not "start_indices" in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
    if len(responsabilities.shape) != 2: raise ValueError("responsabilities must have shape nrotations x npatterns")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    number_of_patterns = len(patterns["start_indices"])-1
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 and scalings.shape[0] == number_of_patterns)):
        raise ValueError("Scalings must have the same shape as responsabilities")
    number_of_rotations = len(slices)
    if scalings is None:
        backend[0].update_slices_sparse(_get_pointer(slices),
                                        number_of_rotations,
                                        _get_pointer(patterns["start_indices"]),
                                        _get_pointer(patterns["indices"]),
                                        _get_pointer(patterns["values"]),
                                        number_of_patterns,
                                        slices.shape[2],
                                        slices.shape[1],
                                        _get_pointer(responsabilities),
                                        resp_threshold)
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        backend[0].update_slices_sparse_scaling(_get_pointer(slices),
                                                number_of_rotations,
                                                _get_pointer(patterns["start_indices"]),
                                                _get_pointer(patterns["indices"]),
                                                _get_pointer(patterns["values"]),
                                                number_of_patterns,
                                                slices.shape[2],
                                                slices.shape[1],
                                                _get_pointer(responsabilities),
                                                resp_threshold,
                                                _get_pointer(scalings))
    else:
        # Scaling per pattern
        backend[0].update_slices_sparse_per_pattern_scaling(_get_pointer(slices),
                                                            number_of_rotations,
                                                            _get_pointer(patterns["start_indices"]),
                                                            _get_pointer(patterns["indices"]),
                                                            _get_pointer(patterns["values"]),
                                                            number_of_patterns,
                                                            slices.shape[2],
                                                            slices.shape[1],
                                                            _get_pointer(responsabilities),
                                                            _get_pointer(scalings))

def calculate_scaling_poisson(patterns, slices, scaling, backend=default_backend):
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
    backend[0].calculate_scaling_poisson(_get_pointer(patterns),
                                         patterns.shape[0],
                                         _get_pointer(slices),
                                         slices.shape[0],
                                         afnumpy.prod(slices.shape[1:]),
                                         _get_pointer(scaling))

def calculate_scaling_per_pattern_poisson(patterns, slices, responsabilities, scaling, backend=default_backend):
    if len(patterns.shape) != 3:
        raise ValueError("Patterns must be a 3D array")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 1:
        raise ValueError("Slices must be a 1D array")
    if len(responsabilities.shape) != 2:
        raise ValueError("Slices must be a 2D array")
    if slices.shape[1:] != patterns.shape[1:]:
        raise ValueError("Slices and patterns must be the same shape")
    if scaling.shape[0] != patterns.shape[0]:
        raise ValueError("scaling must have same length as patterns")
    if slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("Responsabilities must have shape nrotations x npatterns")
    backend[0].calculate_scaling_per_pattern_poisson(_get_pointer(patterns),
                                                     patterns.shape[0],
                                                     _get_pointer(slices),
                                                     slices.shape[0],
                                                     afnumpy.prod(slices.shape[1:]),
                                                     _get_pointer(responsabilities),
                                                     _get_pointer(scaling))

def calculate_scaling_poisson_sparse(patterns, slices, scaling, backend=default_backend):
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
    backend[0].calculate_scaling_poisson_sparse(_get_pointer(patterns["start_indices"]),
                                                _get_pointer(patterns["indices"]),
                                                _get_pointer(patterns["values"]),
                                                number_of_patterns,
                                                _get_pointer(slices),
                                                slices.shape[0],
                                                afnumpy.prod(slices.shape[1:]),
                                                _get_pointer(scaling))

def calculate_scaling_per_pattern_poisson_sparse(patterns, slices, scaling, backend=default_backend):
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
    if len(scaling.shape) != 1:
        raise ValueError("Slices must be a 1D array")
    number_of_patterns = len(patterns["start_indices"])-1
    if scaling.shape[0] != number_of_patterns:
        raise ValueError("scaling must have same length as patterns")
    if slices.shape[0] != responsabilities.shape[0] or number_of_patterns != responsabilities.shape[1]:
        raise ValueError("Responsabilities must have shape nrotations x npatterns")
    backend[0].calculate_scaling_poisson_sparse(_get_pointer(patterns["start_indices"]),
                                                _get_pointer(patterns["indices"]),
                                                _get_pointer(patterns["values"]),
                                                number_of_patterns,
                                                _get_pointer(slices),
                                                slices.shape[0],
                                                afnumpy.prod(slices.shape[1:]),
                                                _get_pointer(responsabilities),
                                                _get_pointer(scaling))

# Attempt to fix for high angles
def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size, edge_distance=None):
    if edge_distance is None:
        edge_distance = image_shape[0]/2.
    x_pixels_1d = afnumpy.arange(image_shape[1]) - image_shape[1]/2. + 0.5
    y_pixels_1d = afnumpy.arange(image_shape[0]) - image_shape[0]/2. + 0.5
    y_pixels, x_pixels = numpy.meshgrid(y_pixels_1d, x_pixels_1d, indexing="ij")
    x_meters = afnumpy.array(x_pixels*pixel_size, dtype=afnumpy.float32)
    y_meters = afnumpy.array(y_pixels*pixel_size, dtype=afnumpy.float32)
    radius_meters = afnumpy.sqrt(x_meters**2 + y_meters**2)

    scattering_angle = afnumpy.arctan(radius_meters / detector_distance)
    z = -1./wavelength*(1. - afnumpy.cos(scattering_angle))
    radius_fourier = afnumpy.sqrt(1./wavelength**2 - (1./wavelength - abs(z))**2)

    x = x_meters * radius_fourier / radius_meters
    y = y_meters * radius_fourier / radius_meters

    x[radius_meters == 0.] = 0.
    y[radius_meters == 0.] = 0.

    output_coordinates = afnumpy.zeros((3, ) + image_shape, dtype=afnumpy.float32)
    output_coordinates[0, :, :] = afnumpy.float32(x)
    output_coordinates[1, :, :] = afnumpy.float32(y)
    output_coordinates[2, :, :] = afnumpy.float32(z)

    # Rescale so that edge pixels match.
    furthest_edge_coordinate = afnumpy.sqrt(x[0, image_shape[1]/2]**2 + y[0, image_shape[1]/2]**2 + z[0, image_shape[1]/2]**2)
    rescale_factor = edge_distance/furthest_edge_coordinate
    output_coordinates *= rescale_factor
    
    return output_coordinates

def ewald_solid_angles(image_shape, detector_distance, pixel_size):
    x_pixels_1d = afnumpy.arange(image_shape[1]) - image_shape[1]/2. + 0.5
    y_pixels_1d = afnumpy.arange(image_shape[0]) - image_shape[0]/2. + 0.5
    y_pixels, x_pixels = numpy.meshgrid(y_pixels_1d, x_pixels_1d, indexing="ij")
    x_meters = afnumpy.array(x_pixels*pixel_size, dtype=afnumpy.float32)
    y_meters = afnumpy.array(y_pixels*pixel_size, dtype=afnumpy.float32)

    distance_meters = afnumpy.sqrt(x_meters**2 + y_meters**2 + detector_distance**2)

    radius_meters = afnumpy.sqrt(x_meters**2 + y_meters**2)
    scattering_angle = afnumpy.arctan(radius_meters / detector_distance)

    solid_angle = pixel_size**2/distance_meters**2*afnumpy.cos(scattering_angle)
    return solid_angle

def rotate_model(model, rotated_model, rotation, backend=default_backend):
    rotation = afnumpy.array(rotation, dtype="float32")
    if model.shape != rotated_model.shape:
        raise ValueError("model and rotated_model must have the same shape")
    if len(rotation.shape) != 1 or rotation.shape[0] != 4:
        raise ValueError("rotation must be length 4 arrray")
    backend[0].rotate_model(_get_pointer(model),
                            _get_pointer(rotated_model),
                            model.shape[2], model.shape[1], model.shape[0],
                            _get_pointer(rotation))    

def chunks(number_of_rotations, chunk_size):
    """Generator for slices to chunk up the data"""
    chunk_starts = afnumpy.arange(0, number_of_rotations, chunk_size)
    chunk_ends = chunk_starts + chunk_size
    chunk_ends[-1] = number_of_rotations
    chunk_sizes = chunk_ends - chunk_starts
    indices_cpu = [slice(this_chunk_start, this_chunk_end) for this_chunk_start, this_chunk_end
                    in zip(chunk_starts, chunk_ends)]
    indices_gpu = [slice(None, this_chunk_end-this_chunk_start) for this_chunk_start, this_chunk_end
                     in zip(chunk_starts, chunk_ends)]
    for this_indices_cpu, this_indices_gpu in zip(indices_cpu, indices_gpu):
        yield this_indices_cpu, this_indices_gpu

def radial_average(image, mask=None):
    """Calculates the radial average of an array of any shape,
    the center is assumed to be at the physical center."""
    if mask is None:
        mask = numpy.ones(image.shape, dtype='bool8')
    else:
        mask = numpy.bool8(mask)
    axis_values = [numpy.arange(l) - l/2. + 0.5 for l in image.shape]
    radius = numpy.zeros((image.shape[-1]))
    for i in range(len(image.shape)):
        radius = radius + (axis_values[-(1+i)][(slice(0, None), ) + (numpy.newaxis, )*i])**2
    radius = numpy.int32(numpy.sqrt(radius))
    number_of_bins = radius[mask].max() + 1
    radial_sum = numpy.zeros(number_of_bins)
    weight = numpy.zeros(number_of_bins)
    for value, this_radius in zip(image[mask], radius[mask]):
        radial_sum[this_radius] += value
        weight[this_radius] += 1.
    radial_sum[weight > 0] /= weight[weight > 0]
    radial_sum[weight == 0] = numpy.nan
    return radial_sum
        
def init_model_radial_average(patterns, randomness=0.):
    """Simple function to create a random start. The new array will have
    a side similar to the second axis of the patterns"""
    pattern_mean = patterns.mean(axis=0)
    pattern_radial_average = radial_average(numpy.array(patterns.mean(axis=0)))
    side = patterns.shape[1]
    x = numpy.arange(side) - side/2 + 0.5

    r_int = numpy.int32(numpy.sqrt(x[:, numpy.newaxis, numpy.newaxis]**2 +
                                   x[numpy.newaxis, :, numpy.newaxis]**2 +
                                   x[numpy.newaxis, numpy.newaxis, :]**2))
    r_int_copy = r_int.copy()
    r_int[r_int >= len(pattern_radial_average)] = 0
    
    model = pattern_radial_average[numpy.int32(r_int)]
    model *= 1. - randomness + 2. * randomness * numpy.random.random((side, )*3)
    model[r_int_copy >= len(pattern_radial_average)] = -1.    
    return afnumpy.array(model, dtype=numpy.dtype("float32"))


def expand_model_2d(model, slices, rotations, backend=default_backend):
    if len(slices) != len(rotations):
        raise ValueError("Slices and rotations must be of the same length.")
    if len(model.shape) != 2:
        raise ValueError("Model must be a 2D array.")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 1:
        raise ValueError("rotations must be a 1D array.")

    number_of_rotations = len(rotations)
    backend[0].expand_model_2d(_get_pointer(model),
                            model.shape[1], model.shape[0],
                            _get_pointer(slices),
                            slices.shape[2], slices.shape[1],
                            _get_pointer(rotations),
                            number_of_rotations)

    
def insert_slices_2d(model, model_weights, slices, slice_weights, rotations, interpolation="linear",
                  backend=default_backend):
    if len(slices) != len(rotations):
        raise ValueError("slices and rotations must be of the same length.")
    if len(slices) != len(slice_weights):
        raise ValueError("slices and slice_weights must be of the same length.")
    if len(slice_weights.shape) != 1:
        raise ValueError("slice_weights must be one dimensional.")
    if len(model.shape) != 2 or model.shape != model_weights.shape:
        raise ValueError("model and model_weights must be 2D arrays of the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 1:
        raise ValueError("Rotations must be a 1D array.")

    interpolation_int = _INTERPOLATION[interpolation]
    number_of_rotations = len(rotations)
    backend[0].insert_slices_2d(_get_pointer(model),
                                _get_pointer(model_weights),
                                model.shape[1], model.shape[0],
                                _get_pointer(slices),
                                slices.shape[2], slices.shape[1],
                                _get_pointer(slice_weights),
                                _get_pointer(rotations),
                                number_of_rotations,
                                interpolation_int)
