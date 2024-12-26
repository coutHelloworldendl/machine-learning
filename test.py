import ctypes
import numpy as np

lib = ctypes.CDLL('./closest.so')
lib.get_closest_point.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.c_double)]

###
# n: number of dimensions 
# G: generator matrix of the lattice (lower-triangular, positive diagonal elements)
# r: point in n-dimensional space
# return: coordinates of the closest point in the lattice to r
###
def get_closest_point(n: int, G: np.ndarray, r: np.ndarray): 
    result = np.zeros(n, dtype=np.int32)

    G_ptr = G.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    r_ptr = r.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    lib.get_closest_point(n, G_ptr, r_ptr, result_ptr)

    return result


###
# n: number of dimensions
# return: n x n matrix with zero-mean, unit-variance Gaussian random numbers
###
def get_gaussian_matrix(n: int):
    return np.random.normal(size=(n, n))

n = 2
G = np.array([[1, 0], [0.5, 1]], dtype=np.float64)
r = np.array([1.5, 1.5], dtype=np.float64)
print(get_closest_point(n, G, r))
