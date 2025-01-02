import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.funcs import gaussian_random as GRAN
from utils.funcs import uniform_random as URAN
from utils.funcs import NSM as NSM
from utils.closest_algo import get_closest_point as CLP
from utils.lll_algo import lll_algorithm as RED
from utils.args import args

# construct a lattice
def construct_lattice(n, f):
    matrix = GRAN([n, n])
    matrix = RED(matrix)
    matrix = np.linalg.cholesky(matrix)
    
    # sanity check
    if np.min([matrix[i][i] for i in range(n)]) <= 0:
        return False, matrix
    v = np.prod([matrix[i][i] for i in range(n)])
    matrix = matrix * pow(v, -1/n)
    
    # main loop
    for t in tqdm(range(args.epoch), desc = 'Constructing lattice'):
        mu = args.mu_0 * pow(args.nu, -t/(args.epoch - 1))
        z = URAN([n])
        y = z - CLP(matrix, z @ matrix)
        e = y @ matrix
        e_2norm = np.linalg.norm(e) ** 2 # squared 2-norm
        for i in range(n):
            for j in range(i):
                matrix[i][j] -= mu * y[i] * e[j]
            matrix[i][i] -= mu * (y[i] * e[i] - e_2norm / (n * matrix[i][i]))
        if t % args.mod == args.mod - 1:
            matrix = np.linalg.cholesky(RED(matrix))
            v = np.prod([matrix[i][i] for i in range(n)])
            matrix = matrix * pow(v, -1/n)
        f.write('Epoch:{}\n'.format(t))
    return True, matrix

if __name__ == '__main__':
    with open(args.log, 'w') as f:
        for i in range(args.try_time):
            status, matrix = construct_lattice(args.n, f)
            if status:
                print('Lattice:{}\n, try_time:{}'.format(matrix, i))
                break
            if i == args.try_time - 1:
                print('Fail to construct a lattice after {} times'.format(args.try_time))
                exit(0)
    plt.matshow(matrix)
    plt.colorbar()
    plt.savefig('lattice.png')    