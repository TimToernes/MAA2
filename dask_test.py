#%%
import dask.array as da
from dask.distributed import Client, progress
import sparse
import numpy as np
import scipy 
import gurobipy as gp
import sys
#%%
client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='2GB')
# %%
x = da.random.random((1000000, 1000000), chunks=(10000, 10000))
x[x < 0.95] = 0
t = x.transpose()
#%%

%time d = x.dot(t)

# %%
s1 = x.map_blocks(sparse.COO)
t1 = s1.transpose()
# %%
%time d1 = s1.dot(t1)
# %%
#import sparse
s2 = x.map_blocks(scipy.sparse.csr_matrix)
t2 = s2.transpose()
# %%
%time d2 = s2.dot(t2)
# %%
m = gp.read('model_small.lp')
#%%
A = da.from_array(m.getA(), chunks=(10000, 10000))

#%%
f = np.random.rand(A.shape[1])*2-1
theta = f/np.cos(f)
# %%

%time A_dot_theta  = A.dot(theta)
# %%

A_spar = m.getA()
# %%

%time A_dot_theta_spar = A_spar.dot(theta)
# %%
sys.getsizeof(A)
# %%
