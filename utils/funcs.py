import numpy as np
from closest_algo import get_closest_point as CLP
from args import args

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
    random_matrix = np.random.rand(args.sample, n)
    result_matrix = np.zeros((args.sample, n))
    for i in range(args.sample):
        result_matrix[i, :] = CLP(n, matrix, random_matrix[i])
    result_matrix -= random_matrix
    e_matrix = result_matrix @ matrix
    row_length = np.linalg.norm(e_matrix, axis=1)
    row_length_square = row_length ** 2
    length_sum = np.sum(row_length_square)
    return np.prod(np.diagonal(matrix)) ** (-2.0 / n) * length_sum / (args.sample * n)

if __name__ == '__main__':
    '''
    for n in range(2, 30):
        array = []
        for _ in range(100):
            matrix = np.array(np.eye(n), dtype = np.float64)
            array.append(NSM(matrix, n))
        print('n = {}, mean = {}, variance = {}'.format(n, np.mean(array), np.var(array)))
    '''
    matrix = np.array([[1, 0, 0], 
                       [0.3639316864866463663, 1.028057868749959747, 0], 
                       [-0.3643806993546727102, 0.5134608145305991078, 0.8911988783086500776]], 
                      dtype = np.float64)
    print(NSM(matrix, 3))
    