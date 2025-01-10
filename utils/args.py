'''
This file contains all the arguments.
Please include all the arguments here.
Use 'from utils.args import args' to import the arguments.
'''

import argparse

parser = argparse.ArgumentParser()

# the dimension of the lattice, n >= 1
parser.add_argument('--n'          , type = list[int]   , default = range(2, 30) ) 
# the number of batches in one epoch
parser.add_argument('--batch_size' , type = int   , default = 8           )
# the number of workers in the thread pool
parser.add_argument('--num_workers', type = int   , default = 4           )
# show the debug information (NSM and matrix) for at most dbg_times times
# if dbg_times = 0, then no debug information will be shown
parser.add_argument('--dbg_times'  , type = int   , default = 0           )
# number of try times when constructing a lattice
parser.add_argument('--try_time'   , type = int   , default = 100         ) 
# max learning rate
parser.add_argument('--mu_0'       , type = float , default = 0.01        ) 
# warm up rate
parser.add_argument('--warm_up'    , type = float , default = 0.1         ) 
# the delta parameter of the LLL algorithm
parser.add_argument('--delta'      , type = float , default = 0.75        ) 
# the decay rate of the learning rate
parser.add_argument('--nu'         , type = float , default = 200         ) 
# number of iterations between two consecutive lattice reduction
parser.add_argument('--mod'        , type = int   , default = 100         )
# log file path
parser.add_argument('--log'        , type = str   , default = './log'     )
#combined record path
parser.add_argument('--record'        , type = str   , default = './record'     )
# number of samples computing NSM for final result
parser.add_argument('--test_sample', type = int   , default = 1000000     )

# the mode of all images 
# 1. save mode: save the image to the path
# 2. show mode: show the image in the window
# 3. empty mode: do nothing, default, if want to draw specific image, set the corresponding mode to 'save' or 'show'
parser.add_argument('--theta_image_mode'  , type = str  , default = 'empty'     )
parser.add_argument('--descend_curve_mode', type = str  , default = 'empty'     )
parser.add_argument('--lattice_graph_mode', type = str  , default = 'empty'     )

# theta image parameters
parser.add_argument('--u_bidirection_range', type = int    , default = 1        )
parser.add_argument('--image_x_upper_bound', type = float  , default = 5        )
parser.add_argument('--sample_num'         , type = float  , default = 500      )

args = parser.parse_args()
for n in args.n:
    assert n >= 1
# run the algorithm for at most epoch times, fast = 100w, mid = 1000w, slow = 10000w
parser.add_argument('--epoch'      , type = int   , default = 1000000 // (args.batch_size * args.num_workers) )
args = parser.parse_args()
# number of samples computing NSM for intermediate debug information, divide by n in main.py
parser.add_argument('--dbg_sample'     , type = int   , default = 1000000     )
# number of iterations between two consecutive debug information
parser.add_argument('--dbg_interval'   , type = int   , default = args.epoch // args.dbg_times if args.dbg_times > 0 else 0)
args = parser.parse_args()