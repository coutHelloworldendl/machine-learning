import argparse

args = argparse.ArgumentParser()
args.add_argument('--n'        , type = int   , default = 3        ) # lattice dimension
args.add_argument('--epoch'    , type = int   , default = 1000000  ) # number of iterations
args.add_argument('--try_time' , type = int   , default = 100      ) # number of tries
args.add_argument('--mu_0'     , type = float , default = 0.005    ) # initial learning rate
args.add_argument('--nu'       , type = float , default = 200      ) # decay rate
args.add_argument('--mod'      , type = int   , default = 100      ) # number of iterations between two consecutive lattice reduction
args.add_argument('--log'      , type = str   , default = '/log'   ) # log file
args = args.parse_args()