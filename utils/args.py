'''
This file contains all the arguments.
Please include all the arguments here.
Use 'from utils.args import args' to import the arguments.
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n'          , type = int   , default = 3           ) # lattice dimension
parser.add_argument('--epoch'      , type = int   , default = 1000000     ) # number of iterations
parser.add_argument('--dbg_times'  , type = int   , default = 20          ) # number of debug information
parser.add_argument('--try_time'   , type = int   , default = 100         ) # number of tries
parser.add_argument('--mu_0'       , type = float , default = 0.005       ) # initial learning rate
parser.add_argument('--delta'      , type = float , default = 0.75        ) # LLL parameter
parser.add_argument('--nu'         , type = float , default = 200         ) # decay rate
parser.add_argument('--mod'        , type = int   , default = 100         ) # number of iterations between two consecutive lattice reduction
parser.add_argument('--log'        , type = str   , default = './log'     ) # log file
args = parser.parse_args()
assert args.n > 0
parser.add_argument('--sample'     , type = int   , default = 1000000 // args.n     ) # number of samples per dimension
parser.add_argument('--dbg_epoch'  , type = int   , default = int(args.epoch / args.dbg_times) ) # number of iterations between two consecutive debug information
args = parser.parse_args()