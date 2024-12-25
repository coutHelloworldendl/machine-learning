import argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('--n'        , type = int   , default = 2       )
args.add_argument('--epoch'    , type = int   , default = 1000000 )
args.add_argument('--mu_0'     , type = float , default = 0.005   )
args.add_argument('--nu'       , type = float , default = 200     )
args.add_argument('--mod'      , type = int   , default = 100     )
args = args.parse_args()

def GRAN(array):
    pass

def URAN(array):
    pass

def RED(matrix):
    pass

def CLP(lattice, vector):
    pass

def construct_lattice(n):
    matrix = GRAN([n, n])
    matrix = RED(matrix)
    matrix = np.linalg.cholesky(matrix)
    if np.min([matrix[i][i] for i in range(n)]) <= 0: # sanity check
        return False, matrix
    v = np.prod([matrix[i][i] for i in range(n)])
    matrix = matrix * pow(v, -1/n)
    for t in range(args.epoch):
        mu = args.mu_0 * pow(args.nu, -t/(args.epoch - 1))
        z = URAN([n])
        y = z - CLP(matrix, z @ matrix)
        e = y @ matrix
        e_2norm = np.linalg.norm(e) ** 2 # squared 2-norm
        for i in range(1, n + 1):
            for j in range(1, n):
                matrix[i][j] -= mu * y[i] * e[j]
            matrix[i][i] -= mu * (y[i] * e[j] - e_2norm / (n * matrix[i][i]))
        if t % args.mod == args.mod - 1:
            matrix = np.linalg.cholesky(RED(matrix))
            v = np.prod([matrix[i][i] for i in range(n)])
            matrix = matrix * pow(v, -1/n)
    return True, matrix

def NSM(matrix, n):
    pass

if __name__ == '__main__':
    for i in range():
        status, matrix = construct_lattice(args.n)
        if status:
            print('Lattice:\n, fail_time:'.format(matrix, i))
            break
    