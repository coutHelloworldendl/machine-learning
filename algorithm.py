import argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('--n'        , type = int   , default = 4       )
args.add_argument('--epoch'    , type = int   , default = 1000000 )
args.add_argument('--try_time' , type = int   , default = 100     )
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
        print(e)
        e_2norm = np.linalg.norm(e) ** 2 # squared 2-norm
        print(e_2norm)
        for i in range(n):
            for j in range(i):
                matrix[i][j] -= mu * y[i] * e[j]
            matrix[i][i] -= mu * (y[i] * e[i] - e_2norm / (n * matrix[i][i]))
        if t % args.mod == args.mod - 1:
            matrix = np.linalg.cholesky(RED(matrix))
            v = np.prod([matrix[i][i] for i in range(n)])
            matrix = matrix * pow(v, -1/n)
    return True, matrix

def NSM(matrix, n):
    pass

if __name__ == '__main__':
    for i in range(args.try_time):
        status, matrix = construct_lattice(args.n)
        if status:
            print('Lattice:{}\n, try_time:{}'.format(matrix, i))
            break
        if i == args.try_time - 1:
            print('Fail to construct a lattice after {} times'.format(args.try_time))
            exit(0)
    print('Loss:{}'.format(NSM(matrix, args.n)))