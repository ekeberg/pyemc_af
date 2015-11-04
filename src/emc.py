import emc_cuda
import afnumpy

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
    if len(slices.shape) != 3: raise ValueError("patterns must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    #sigma = afnumpy.float32(sigma)
    if sigma <= 0.: raise ValueError("sigma must be larger than zeros")
    patterns_pointer = emc_cuda.int_to_float_pointer(patterns.d_array.device_ptr())
    slices_pointer = emc_cuda.int_to_float_pointer(slices.d_array.device_ptr())
    responsabilities_pointer = emc_cuda.int_to_float_pointer(responsabilities.d_array.device_ptr())
    emc_cuda.cuda_calculate_responsabilities(patterns_pointer, patterns.shape[0], slices_pointer, slices.shape[0],
                                             slices.shape[1], slices.shape[2], responsabilities_pointer, sigma)

    
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
