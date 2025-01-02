import ctypes
import numpy as np

lib = ctypes.CDLL('./utils/lll.so')
lib.lll_algorithm.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_double]

def lll_algorithm(G: np.ndarray, n: int, delta: float):
    G_ptr = G.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    lib.lll_algorithm(n, G_ptr, delta) 
    return G

def test():
    n = 3
    G = np.array([[1, 1, 1], [-1, 0, 2], [3, 5, 6]], dtype=np.float64)
    print(lll_algorithm(G, n, 0.75))

if __name__ == '__main__':
    test()