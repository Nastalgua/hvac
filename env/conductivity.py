import numpy as np

KERNAL = np.matrix([1.0 / 9.0]).repeat(9).reshape(3, 3)
KERNAL[1, 1] = (1.0 / 9.0) - 1

OUTSIDE_TEMP = 90

def conv2d(input_matrix: np.ndarray, mask: np.ndarray):
    view_shape = mask.shape + tuple(np.subtract(input_matrix.shape, mask.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    sub_matrix = strd(input_matrix, shape=view_shape, strides=input_matrix.strides * 2)

    return np.einsum('ij,ijkl->kl', mask, sub_matrix)

def apply_conductivity(curr_temps: np.ndarray, thermal_conductivity: np.ndarray):
    delta = conv2d(input_matrix=curr_temps, mask=KERNAL)
    new_delta = np.multiply(delta, thermal_conductivity)
    new_curr_temps = curr_temps[1:-1, 1:-1]

    return np.pad(new_curr_temps + new_delta, 1, constant_values=[OUTSIDE_TEMP])
