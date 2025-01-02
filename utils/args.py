'''
This file contains all the arguments.
Please include all the arguments here.
Use 'from utils.args import args' to import the arguments.
'''

import argparse

args = argparse.ArgumentParser()
args.add_argument('--n'        , type = int   , default = 3           ) # lattice dimension
args.add_argument('--epoch'    , type = int   , default = 1000000     ) # number of iterations
args.add_argument('--dbg_epoch', type = int   , default = 10000       ) # number of iterations between two consecutive debug information
args.add_argument('--try_time' , type = int   , default = 100         ) # number of tries
args.add_argument('--mu_0'     , type = float , default = 0.005       ) # initial learning rate
args.add_argument('--delta'    , type = float , default = 0.75        ) # LLL parameter
args.add_argument('--nu'       , type = float , default = 200         ) # decay rate
args.add_argument('--mod'      , type = int   , default = 100         ) # number of iterations between two consecutive lattice reduction
args.add_argument('--log'      , type = str   , default = './log'     ) # log file
args = args.parse_args()