import argparse

args = argparse.ArgumentParser()
args.add_argument('--n'        , type = int   , default = 4        )
args.add_argument('--epoch'    , type = int   , default = 1000000  )
args.add_argument('--try_time' , type = int   , default = 100      )
args.add_argument('--mu_0'     , type = float , default = 0.005    )
args.add_argument('--nu'       , type = float , default = 200      )
args.add_argument('--mod'      , type = int   , default = 100      )
args.add_argument('--log'      , type = str   , default = '/log'   )
args = args.parse_args()