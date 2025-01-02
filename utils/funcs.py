import numpy as np

# sample from a normal distribution
def gaussian_random(array):
    return np.random.normal(size = array)

# sample from a uniform distribution
def uniform_random(array):
    return np.random.uniform(low = 0, high = 1, size = array)

# sanity check
def sanity_check(matrix, n):
    if np.min([matrix[i][i] for i in range(n)]) <= 0:
        return False
    else:
        return True

# test the result lattice
def NSM(matrix, n):

    '''
    Your code here
    '''
    
    pass