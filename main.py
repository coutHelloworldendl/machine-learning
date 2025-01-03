import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.funcs import gaussian_random as GRAN
from utils.funcs import uniform_random as URAN
from utils.funcs import orthogonalize as ORTH
from utils.funcs import NSM as NSM
from utils.funcs import sanity_check as SC
from utils.closest_algo import get_closest_point as CLP
from utils.lll_algo import lll_algorithm as RED
from utils.args import args

# construct a lattice
def construct_lattice(n, f):
    # generate a random matrix
    matrix = GRAN([n, n]) 
    
    # reduce the matrix using LLL algorithm
    matrix = RED(matrix, n=n, delta=args.delta)
    
    # orthogonalize the matrix
    matrix = ORTH(matrix)
    
    # normalize the matrix
    v = np.prod([matrix[i][i] for i in range(n)])
    matrix = matrix * pow(v, -1.0/n)
    
    # main loop
    for t in tqdm(range(args.epoch), desc = 'Constructing lattice'):
        mu = args.mu_0 * pow(args.nu, -t/(args.epoch - 1))
        z = URAN([n])
        y = z - CLP(n, matrix, z @ matrix)
        e = y @ matrix
        e_2norm = np.linalg.norm(e) ** 2 # squared 2-norm
        for i in range(n):
            for j in range(i):
                matrix[i][j] -= mu * y[i] * e[j]
            matrix[i][i] -= mu * (y[i] * e[i] - e_2norm / (n * matrix[i][i])) # gradient descent
        result = SC(matrix, n) # sanity check
        if not result:
            f.write('Fail to construct a lattice\n')
            return False, None
        if t % args.mod == args.mod - 1:
            matrix = RED(matrix, n=n, delta=args.delta)
            matrix = ORTH(matrix)
            v = np.prod([matrix[i][i] for i in range(n)])
            matrix = matrix * pow(v, -1/n)
        if (t + 1) % args.dbg_epoch == 0:
            f.write('Epoch {}: matrix = {}\n'.format(t + 1, matrix))
    return True, matrix

if __name__ == '__main__':
    
    # create log file path
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    
    # create log file
    log_path = args.log + '/log-dim-' + str(args.n) + '.txt'
    result_path = args.log + '/result-dim-' + str(args.n) + '.txt'
    graph_path = args.log + '/graph-dim-' + str(args.n) + '.png'
        
    # construct a lattice
    with open(log_path, 'w') as f:
        for i in range(args.try_time):
            
            # try to construct a lattice
            status, matrix = construct_lattice(args.n, f)
            for i in range(args.n):
                matrix[i] /= np.linalg.norm(matrix[i])
            
            # if success to construct a lattice, break the loop
            if status:
                print('Lattice:\n {}\nTry_time:{}'.format(matrix, i))
                break
            
            # if fail to construct a lattice, try again, up to args.try_time times
            if i == args.try_time - 1:
                print('Fail to construct a lattice after {} times'.format(args.try_time))
                exit(0)
    
    # save and visualize the lattice
    with open(result_path, 'w') as f:
        f.write('Lattice =\n{}\n'.format(matrix))
    plt.matshow(matrix)
    plt.colorbar()
    plt.savefig(graph_path)