'''
This file contains all the arguments.
Please include all the arguments here.
Use 'from utils.args import args' to import the arguments.
'''

import argparse

parser = argparse.ArgumentParser()

# the dimension of the lattice, n >= 1
parser.add_argument('--n'          , type = int   , default = 3           ) 
# run the algorithm for at most epoch times, fast = 100w, mid = 1000w, slow = 10000w
parser.add_argument('--epoch'      , type = int   , default = 1000000     ) 
# show the debug information (NSM and matrix) for at most dbg_times times
# if dbg_times = 0, then no debug information will be shown
parser.add_argument('--dbg_times'  , type = int   , default = 0           )
# number of try times when constructing a lattice
parser.add_argument('--try_time'   , type = int   , default = 100         ) 
# max learning rate
parser.add_argument('--mu_0'       , type = float , default = 0.01       ) 
# warm up rate
parser.add_argument('--warm_up'       , type = float , default = 0.1      ) 
# the delta parameter of the LLL algorithm
parser.add_argument('--delta'      , type = float , default = 0.75        ) 
# the decay rate of the learning rate
parser.add_argument('--nu'         , type = float , default = 200         ) 
# number of iterations between two consecutive lattice reduction
parser.add_argument('--mod'        , type = int   , default = 100         )
# log file path
parser.add_argument('--log'        , type = str   , default = './log'     )
# number of samples computing NSM for final result
parser.add_argument('--test_sample'    , type = int   , default = 1000000 )

args = parser.parse_args()
assert args.n > 0 

# number of samples computing NSM for intermediate debug information
parser.add_argument('--dbg_sample'     , type = int   , default = 1000000 // args.n     )
# number of iterations between two consecutive debug information
parser.add_argument('--dbg_epoch'  , type = int   , default = int(args.epoch / args.dbg_times) if args.dbg_times > 0 else 0)

args = parser.parse_args()