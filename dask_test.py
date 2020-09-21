#%%
import dask.array as da
from dask.distributed import Client, progress
import sparse
import numpy as np
import scipy 
import scipy.linalg
import scipy.sparse.linalg
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
#%% test for computing null space of sparse matrix 

A = np.random.rand(50,50)
A[A<0.95] = 0 
#A[range(len(A)),range(len(A))] = 1

A_spar = scipy.sparse.csr_matrix(A)

#%%
A_spar = scipy.sparse.rand(10000,5000,0.01)
#%%
print('rank of A {}'.format(np.linalg.matrix_rank(A)))
print('null matrix shape {}x{}'.format(A.shape[1],A.shape[1]-np.linalg.matrix_rank(A)))

#%% Method 1
N1 = scipy.linalg.null_space(A)

# %% Method 2
# of the singular values/vectors is probably not feasible for large sparse matrices
u, s, vh = np.linalg.svd(A, full_matrices=False)
#tol = np.finfo(A.dtype).eps * A.nnz
k = A.shape[1]-np.linalg.matrix_rank(A)
s_copy = s.copy()
s_copy.sort()
tol = s_copy[k-1]

N2 = vh.compress(s <= tol, axis=0).conj().T
# %% Method 3

k = A.shape[1]-np.linalg.matrix_rank(A)
u,s,vt = scipy.sparse.linalg.svds(A_spar,
                                  k=k,
                                  which='SM',
                                  tol=1e-12)
N3 = vt.T

#%% Method 4 

Q, _, _,r = sparseqr.qr( A_spar.transpose() )
del _ 
N4 = Q.tocsr()[:,r:]


#%% Method 5

eps = 1e-12
u, s, vh = scipy.sparse.linalg.svds(A_spar,k=50)
padding = max(0,np.shape(A)[1]-np.shape(s)[0])
null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
null_space = scipy.compress(null_mask, vh, axis=0)

# %%

print(np.max(np.dot(A,N1)))
print(np.max(np.dot(A,N2)))
print(np.max(np.dot(A,N3)))
print(A_spar.dot(N4).max())
# %%


def test():
    return 1,2,3,4
# %%

def spar_zeros(n,m):
    zeros = scipy.sparse.csr_matrix((np.array([]),(np.array([]),np.array([]))),shape=[n,m])
    return zeros
# %%
def null(A, eps=1e-12):
...    u, s, vh = scipy.linalg.svd(A)
...    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
...    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
...    null_space = scipy.compress(null_mask, vh, axis=0)
...    return scipy.transpose(null_space)