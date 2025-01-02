import ctypes
import numpy as np

lib = ctypes.CDLL('./lll.so')
lib.lll_algorithm.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_double]

###
# n: number of dimensions 
# G: generator matrix of the lattice (lower-triangular, positive diagonal elements)
# r: point in n-dimensional space
# return: coordinates of the closest point in the lattice to r
###

def lll_algorithm(G: np.ndarray, n: int, delta: float):
    G_ptr = G.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    lib.lll_algorithm(G_ptr, n, delta) 
    return G

def test():
    n = 3
    G = np.array([[1, 1, 1], [-1, 0, 2], [3, 5, 6]], dtype=np.float64)
    print(lll_algorithm(G, n, 0.75))

if __name__ == '__main__':
    test()