import numpy as np
from utils.closest_algo import get_closest_point as CLP


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
    
# cholosky decomposition
def orthogonalize(matrix):
    matrix = matrix @ matrix.T
    matrix = np.linalg.cholesky(matrix)
    return matrix

# test the result lattice
def NSM(matrix, n):
    t = 10
    random_point_num = t**n
    random_matrix = np.random.rand(random_point_num, n)
    result_matrix = np.zeros((random_point_num, n))
    for i in range(random_point_num):
        result_matrix[i, :] = CLP(n, matrix, random_matrix[i])
    result_matrix-=random_matrix
    e_matrix = result_matrix@matrix
    row_length = np.linalg.norm(e_matrix, axis=1)
    row_length_square = row_length**2
    length_sum = np.sum(row_length_square)
    volume=1
    for i in range(n):
        volume*=matrix[i][i]
    return volume**(-2/n)*length_sum/random_point_num