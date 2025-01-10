import numpy as np
from .closest_algo import get_closest_point as CLP
from .args import args

# sample from a normal distribution
def gaussian_random(array):
    return np.random.normal(size = array)

# sample from a uniform distribution
def uniform_random(array):
    return np.random.uniform(low = 0, high = 1, size = array)

# sanity check
def sanity_check(matrix):
    return np.min(np.diag(matrix)) > 0
    
# cholosky decomposition
def orthogonalize(matrix):
    matrix = matrix @ matrix.T
    matrix = np.linalg.cholesky(matrix)
    return matrix

# test the result lattice
def NSM(matrix, n, sample):
    random_matrix = np.random.rand(sample, n)
    random_matrix = random_matrix @ matrix
    result_matrix = np.zeros((sample, n))
    for i in range(sample):
        result_matrix[i, :] = CLP(n, matrix, random_matrix[i])
    result_matrix = random_matrix - result_matrix @ matrix
    row_length_square = np.linalg.norm(result_matrix, axis=1) ** 2
    length_sum = np.mean(row_length_square)
    return (np.prod(np.diagonal(matrix)) ** (-2.0 / n)) * length_sum / n

def read_lattice(filename):
    lattice = []
    nsm = None
    with open(filename, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('Lattice'):
            start_index = i + 1
        elif line.strip().startswith('NSM'):
            end_index = i - 1
            break
    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index + 1]:
            row = line.strip().replace('[', '').replace(']', '').replace(',', '').split()
            lattice.append([float(x) for x in row])
    
    return lattice

def write_lattice(n):
    # Initialize with low-dimensional basis
    # thus, steps to ensure matrix a positive definite matrix 
    # and decompose the matrix into a lower triangular matrix
    # are useless
    n1 = n // 2
    n2 = n - n1
    filename1 = 'record/result-dim-' + str(n1) + '.txt'
    filename2 = 'record/result-dim-' + str(n2) + '.txt'
    lattice1 = read_lattice(filename1)
    lattice2 = read_lattice(filename2)

    matrix = np.zeros((n, n))
    matrix[:n1,:n1] = lattice1
    matrix[n1:,n1:] = lattice2
    return matrix
    
    
if __name__ == '__main__':
    matrix = np.array([[1, 0, 0], 
                       [0.3639316864866463663, 1.028057868749959747, 0], 
                       [-0.3643806993546727102, 0.5134608145305991078, 0.8911988783086500776]],
                      dtype = np.float64)
    print("NSM = {}".format(NSM(matrix, 3, args.test_sample)))
    matrix = np.array([[1, 0, 0], 
                       [-0.5, np.sqrt(3.0)/2.0, 0], 
                       [-0.5, np.sqrt(3.0)/6.0, np.sqrt(2.0/3.0)]],
                      dtype = np.float64)
    print("NSM = {}".format(NSM(matrix, 3, args.test_sample)))
    matrix = np.array([[1, 0, 0], 
                       [0, 1, 0],
                       [0, 0, 1]],dtype=np.float64)
    print("NSM = {} , std = {}".format(NSM(matrix, 3, args.test_sample), 1.0 / 12))
    matrix = np.array([[2, 0, 0], 
                       [0, 2, 0],
                       [0, 0, 2]], dtype=np.float64)
    print("NSM = {} , std = {}".format(NSM(matrix, 3, args.test_sample), 1.0 / 12))
    '''
    for n in range(2, 30):
        array = []
        for _ in range(100):
            matrix = np.array(np.eye(n), dtype = np.float64)
            array.append(NSM(matrix, n))
        print('n = {}, mean = {}, variance = {}'.format(n, np.mean(array), np.var(array)))
    '''