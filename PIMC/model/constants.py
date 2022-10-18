import numpy as np

NT = 10000
BETA = 10
ETA = BETA/NT
DELTA_METRO=0.5
PATHS=1000000
TERM=10000
NT_ARRAY=np.arange(100, 450, 50)
ETA_ARRAY = BETA/(np.array(NT_ARRAY))
N_BINS=10
BIN_INIT=200
# max_size=189000
# block_step=500
BIN_ARRAY=np.array([BIN_INIT*2**i for i in range(0, N_BINS)])
# BIN_ARRAY=np.array([BIN_INIT*i for i in range(1, N_BINS)])
# BIN_ARRAY=np.arange(1, max_size, block_step)
# print((BIN_ARRAY.max()))
