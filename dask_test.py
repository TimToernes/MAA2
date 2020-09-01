#%%
import dask.array as da
import sparse
import numpy as np
# %%
x = da.random.random((100000, 100000), chunks=(1000, 1000))
x[x < 0.95] = 0
# %%
s = x.map_blocks(sparse.COO)
# %%

t = s.transpose()
# %%
%time d = s.dot(t)
# %%
