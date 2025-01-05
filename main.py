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
    # record NSM
    NSM_array = []
    
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

        prod = np.expand_dims(y, axis=1) @ np.expand_dims(e, axis=0)
        matrix -= np.tril(mu * prod)
        matrix += np.eye(n) * mu * (e_2norm / (n * np.diag(matrix)))

        result = SC(matrix, n) # sanity check
        if not result:
            f.write('Fail to construct a lattice\n')
            return False, None, None
        if t % args.mod == args.mod - 1:
            matrix = RED(matrix, n=n, delta=args.delta)
            matrix = ORTH(matrix)
            v = np.prod(np.diag(matrix))
            matrix = matrix * pow(v, -1/n)
        if (t + 1) % args.dbg_epoch == 0:
            nsm = NSM(matrix, n)
            f.write('Epoch = {}, NSM = {}, matrix =\n{}\n'.format(t + 1, nsm, matrix))
            NSM_array.append(nsm)
    return True, matrix, NSM_array

if __name__ == '__main__':
    # create log file path
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    
    # create log file
    log_path = args.log + '/log-dim-' + str(args.n) + '.txt'
    result_path = args.log + '/result-dim-' + str(args.n) + '.txt'
    graph_path = args.log + '/graph-dim-' + str(args.n) + '.png'
    curve_path = args.log + '/curve-dim-' + str(args.n) + '.png'
        
    # construct a lattice
    with open(log_path, 'w') as f:
        # training args
        f.write('args = {}\n'.format(args))
        
        for i in range(args.try_time):
            
            # try to construct a lattice
            status, matrix, array = construct_lattice(args.n, f)
            
            # normalize the matrix
            for j in range(args.n):
                matrix[j] /= np.linalg.norm(matrix[j])
            
            # if success to construct a lattice, break the loop
            if status:
                print('Lattice:\n{}\nTry_time:{}'.format(matrix, i))
                break
            
            # if fail to construct a lattice, try again, up to args.try_time times
            if i == args.try_time - 1:
                print('Fail to construct a lattice after {} times'.format(args.try_time))
                exit(0)
    
    # save and visualize the lattice
    with open(result_path, 'w') as f:
        f.write('Lattice =\n{},\nNSM =\n{}'.format(matrix, array[-1]))
        
    # visualize the matrix
    plt.matshow(matrix)
    plt.colorbar()
    plt.savefig(graph_path)
    
    # visualize the curve
    x = [i for i in range(1, args.epoch + 1, args.dbg_epoch)]
    plt.clf()
    plt.plot(x, array)
    plt.savefig(curve_path)