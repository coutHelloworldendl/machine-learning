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

# Adam optimizer
class Adam:
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.reset()
    
    # the adam optimizer algorithm
    def step(self, g: np.ndarray):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.linalg.norm(g)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return m_hat / (v_hat + self.epsilon)
    
    # reset the optimizer
    def reset(self):
        self.m, self.v, self.t = 0, 0, 0

# Learning rate scheduler
class Scheduler:
    def __init__(self, args):
        self.lr_initial, self.lr_max, self.nu, self.epoch, self.warm_up = args.mu_0 / args.nu, args.mu_0, args.nu, args.epoch, args.warm_up
    
    # when less than the threshold, the learning rate is linearly increased from lr_initial to lr_max
    # when greater than the threshold, the learning rate is exponentially decayed with decay rate nu
    def step(self, t):
        threshold = self.epoch * self.warm_up
        if t < threshold:
            return self.lr_initial + (self.lr_max - self.lr_initial) * t / (self.epoch * self.warm_up)
        else: 
            return self.lr_max * pow(self.nu, -(t - threshold) / (self.epoch - threshold))
            

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

    # initialize the optimizer and scheduler
    optimizer = Adam()
    scheduler = Scheduler(args)
    
    # main loop
    for t in tqdm(range(args.epoch), desc = 'Constructing lattice'):
        
        # set the learning rate according to epoch
        mu = scheduler.step(t)
        
        z = URAN([n])
        y = z - CLP(n, matrix, z @ matrix)
        e = y @ matrix
        e_2norm = np.linalg.norm(e) ** 2 # squared 2-norm

        prod = np.expand_dims(y, axis=1) @ np.expand_dims(e, axis=0)
        grad = np.tril(prod) - np.eye(n) * (e_2norm / (n * np.diag(matrix)))

        # update the matrix
        matrix -= mu * optimizer.step(grad)

        result = SC(matrix, n) # sanity check
        if not result:
            f.write('Fail to construct a lattice\n')
            return False, None, None
        if t % args.mod == args.mod - 1:
            optimizer.reset()
            matrix = RED(matrix, n=n, delta=args.delta)
            matrix = ORTH(matrix)
            v = np.prod(np.diag(matrix))
            matrix = matrix * pow(v, -1/n)
        if args.dbg_epoch > 0 and (t + 1) % args.dbg_epoch == 0:
            nsm = NSM(matrix, n, args.dbg_sample // n)
            f.write('Epoch = {}, NSM = {}, matrix =\n{}\n'.format(t + 1, nsm, matrix))
            NSM_array.append(nsm)
    return True, matrix, NSM_array

if __name__ == '__main__':
    
    # print all the elements in the matrix
    np.set_printoptions(threshold=np.inf)
    
    # create log file path
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    
    for n in args.n:
    
        # create log file
        log_path = args.log + '/log-dim-' + str(n) + '.txt'
        result_path = args.log + '/result-dim-' + str(n) + '.txt'
        graph_path = args.log + '/graph-dim-' + str(n) + '.png'
        curve_path = args.log + '/curve-dim-' + str(n) + '.png'
        
        # construct a lattice
        with open(log_path, 'w') as f:
            # training args
            f.write('args = {}\n'.format(args))
        
            for i in range(args.try_time):
            
                # try to construct a lattice
                status, matrix, array = construct_lattice(n, f)
            
                # normalize the matrix
                for j in range(n):
                    matrix[j] /= np.linalg.norm(matrix[j])
            
                # if success to construct a lattice, break the loop
                if status:
                    print('Success to construct a lattice after {} times'.format(i + 1))
                    break
            
                # if fail to construct a lattice, try again, up to args.try_time times
                if i == args.try_time - 1:
                    print('Fail to construct a lattice after {} times'.format(args.try_time))
                    exit(0)
    
        # evaluate the lattice
        print('Evaluate the lattice:', end=' ')
        nsm = NSM(matrix, n, args.test_sample)
        print('NSM = {}'.format(nsm))
    
        # save the result
        with open(result_path, 'w') as f:
            f.write('Lattice =\n{},\nNSM =\n{}'.format(matrix, nsm))
        
        # visualize the matrix
        plt.matshow(matrix)
        plt.colorbar()
        plt.savefig(graph_path)
    
        # visualize the curve
        if args.dbg_epoch > 0:
            x = np.arange(1, args.epoch + 1, args.dbg_epoch)
            plt.clf()
            plt.plot(x, array)
            plt.savefig(curve_path)
