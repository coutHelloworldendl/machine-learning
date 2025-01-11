import numpy as np
import matplotlib.pyplot as plt
import math
from .args import args

# the result of our 10-dimension lattice
dim_10_lattice = np.array([[ 1.         , 0.        ,  0.         , 0.   ,       0.       ,   0.,
   0.      ,    0.     ,     0.       ,   0.        ],
 [ 0.47608926 , 0.87939696 , 0.      ,    0.      ,    0.     ,     0.,
   0.      ,    0.        ,  0.     ,     0.        ],
 [ 0.48983154 , 0.30919513  ,0.81514627 , 0.     ,     0.   ,       0.,
   0.    ,      0.      ,    0.    ,      0.        ],
 [ 0.49490372, -0.28153435,  0.40909556 , 0.71305648 , 0.      ,    0.,
   0.       ,   0.        ,  0.     ,     0.        ],
 [-0.4805966 , -0.31103074, -0.2058742,  -0.34645431 , 0.71405323 , 0. ,
   0.      ,    0.       ,   0.      ,    0.        ],
 [ 0.48051727 , 0.30973167 , 0.21552451  ,0.34449037 ,-0.00382895 , 0.71276249,
   0.    ,      0.  ,        0.    ,      0.        ],
 [-0.44603043 ,-0.25822885, -0.16735001 , 0.30824371, -0.31671167  ,0.32034575,
   0.6390828 ,  0.       ,   0.     ,     0.        ],
 [ 0.44210612,  0.2743698 , -0.35221255, -0.01122988, -0.31826547,  0.31850218,
   0.30673526 , 0.55521215 , 0.         , 0.        ],
 [ 0.44429459 ,-0.27244392 , 0.3511344  , 0.00354484,  0.31759695 ,-0.31481628,
  -0.30828321,  0.1759832 ,  0.52828303 , 0.        ],
 [ 0.47430944,  0.30212574,  0.19662104 , 0.35315954, -0.0075109  , 0.01454398,
  -0.35134549, -0.20876238 , 0.28377714,  0.52205968]])

# how many numbers are <= target in a sorted array arr
def bin_search(arr, target):
    l, r = 0, len(arr) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if arr[mid] <= target:
            l = mid
        else:
            r = mid - 1
    return l + (1 if target >= arr[0] else 0)

# 1. run u for every dimension in u_bidirection_range 
# 2. calculate the norm of the lattice.T @ u within the range of image_x_upper_bound
# 3. store the result in store_array
def theta_image_dfs(lattice, pos, dim, store_array:list[int], u_array, u_bidirection_range:list[int], image_x_upper_bound):
    if pos == dim:
        result = np.linalg.norm(lattice.T @ u_array) ** 2
        if result <= image_x_upper_bound:
            store_array.append(result)
        return
    else: 
        for i in range(-u_bidirection_range, u_bidirection_range + 1):
            u_array[pos] = i
            theta_image_dfs(lattice, pos + 1, dim, store_array, u_array, u_bidirection_range, image_x_upper_bound)   
        
# draw the theta image of the lattice
# - u_bidirection_range: the range of u in each dimension, e.g. if u_bidirection_range = 1, then u = {-1, 0, 1}
# - image_x_upper_bound: the upper bound of the x-axis in the image
# - sample_num: the number of dots in the x-axis
# - save mode: save the image to the path
# - show mode: show the image
# - empty mode: do nothing
def draw_theta_image(lattice, u_bidirection_range, image_x_upper_bound:float, sample_num:float, mode='save'):
    # sanity check of attributes
    if mode not in ['save', 'show', 'empty', 'dot']:
        raise ValueError("theta image mode should be 'save' or 'show' or 'empty'")
    if mode == 'empty':
        return None
    assert len(lattice.shape) == 2
    assert lattice.shape[0] == lattice.shape[1]
    
    # get the dimension of the lattice
    dim = lattice.shape[0]
    
    # store the distance of each |uB|
    distance_array = []
    
    # initialize the u array
    u_array = [0] * dim
    
    # normalize the lattice
    V = np.linalg.det(matrix)
    matrix = matrix * (V ** (-1.0 / dim))
    
    # run the dfs, store the result in distance_array
    theta_image_dfs(lattice, 0, dim, distance_array, u_array, u_bidirection_range, image_x_upper_bound)
    
    # sort the distance_array
    distance_array.sort()
    
    # how the dots are distributed among the x-axis
    x_dot_range = np.arange(0, image_x_upper_bound, image_x_upper_bound / sample_num)
    
    # initialize the plot
    plt.clf()
    
    # set labels
    plt.xlabel('r^2')
    plt.ylabel('log10(N(B,r))')
    
    # plot the dots
    plt.plot(x_dot_range, [math.log10(bin_search(distance_array, x)) for x in x_dot_range], linestyle='--', marker='.')
    
    if mode == 'dot':
        path = args.log + '/theta_img_dim-' + str(dim) + '.txt'
        with open(path, 'w') as f:
            for i in range(len(x_dot_range)):
                f.write('{:10f} {:2}\n'.format(x_dot_range[i], int(bin_search(distance_array, x_dot_range[i]))))
    
    # save or show the plot
    elif mode == 'save' :
        path = args.log + '/theta_img_dim-' + str(dim) + '.png'
        plt.savefig(path)
    elif mode == 'show' :
        plt.show()

# draw descend curve
def draw_descend_curve(array, n, mode):
    if mode not in ['save', 'show', 'empty']:
        raise ValueError("descend curve mode should be 'save' or 'show' or 'empty'")
    if mode == 'empty' or args.dbg_interval <= 0 :
        return
    plt.clf()
    path = args.log + '/curve-dim-' + str(n) + '.png'
    x = np.arange(1, args.epoch + 1, args.dbg_interval)
    plt.plot(x, array)
    if mode == 'show':
        plt.show()
    else:
        plt.savefig(path)
            
# draw lattice graph
def draw_lattice(matrix, n, mode):
    if mode not in ['save', 'show', 'empty']:
        raise ValueError("draw mode should be 'save' or 'show' or 'empty'")
    if mode == 'empty':
        return
    plt.clf()
    path = args.log + '/graph-dim-' + str(n) + '.png'
    plt.matshow(matrix)
    plt.colorbar()
    if mode == 'show':
        plt.show()
    else:
        plt.savefig(path)

if __name__ == '__main__':
    
    # sanity check of "bin_search"
    print("bin_search_out:", end='')
    for i in [0.8, 1.2, 3., 3.1, 4. , 5., 5.5]:
        print(bin_search([1., 2., 3., 4., 5.], i), end=',')
    print("\nbin_search_ans:", [0, 1, 3, 3, 4, 5, 5])
    
    # sanity check of "draw_theta_image"
    draw_theta_image(lattice=dim_10_lattice, 
                     u_bidirection_range=1, 
                     image_x_upper_bound=5,
                     sample_num=500,
                     mode='show')