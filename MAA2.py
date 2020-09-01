#%%
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import pypsa
from pyomo.core import ComponentUID
import gurobipy as gp
import scipy 
from scipy import stats
import scipy.optimize
import numpy as np
from sample import sample as rand_walk_sample
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pyomo.environ as pyomo_env
import pickle 
from gurobipy import GRB


# %% Function definitions 

def presolve(A,b,sense):
    # The presolve algorithm will find a fully dimensional sub problem to the original problem
    # given in A. 
    A,b,H,c = step1(A,b,sense)
    A,b,H,c = step2(A,b,H,c)
    A,b,N,x_0 = step3(A,b,H,c)

    return A,b,N,x_0


def step1(A,b,sense):
    # Step 1 - Sorting of the raw A array 
    # Empty rows are removed
    # > constraints are fliped to <
    # = constraints are moved from the A to the H array
    b_1 = []
    H_1 = []
    c_1 = []
    rows_to_delete = []

    indices = A.indices
    indptr = A.indptr

    for id_row in range(len(b)):
        # all elements in row == 0, remove row
        filt = indices[indptr[id_row]:indptr[id_row+1]]
        if len(filt)==0:
            rows_to_delete.append(id_row)
        elif sense[id_row] == '>':
            A[id_row,filt] = -A[id_row,filt]
            b_1.append(-b[id_row])
        elif sense[id_row] == '<':
            pass
            #A_new.append(A[id_row,:])
            b_1.append(b[id_row])
        elif sense[id_row] == '=':
            H_1.append(A.getrow(id_row).toarray()[0])
            c_1.append(b[id_row])
            rows_to_delete.append(id_row)

    delete_rows_csr(A,rows_to_delete[::-1])
    b_1 = np.array(b_1) 
    print('step 1 done ')
    return A,b_1,H_1,c_1

def step2_1(A):
    # Step 2.1 - Finding all constraints in A containing only one variable 
    # And saving this value as an upper or lower bound
    indices = A.indices
    indptr = A.indptr
    ub_idx = []
    ub_idb = []
    lb_idx = []
    lb_idb = []
    for id_row in range(A.shape[0]):
        filt = indices[indptr[id_row]:indptr[id_row+1]]
        if len(filt)==1:
            if A[id_row,filt][0]>0:
                ub_idx.append(filt[0])
                ub_idb.append(id_row)
            elif A[id_row,filt][0]<0:
                lb_idx.append(filt[0])
                lb_idb.append(id_row)

    ub_idx = np.array(ub_idx)
    ub_idb = np.array(ub_idb)
    lb_idx = np.array(lb_idx)
    lb_idb = np.array(lb_idb)
    print('step 2.1 done')
    return ub_idx,ub_idb,lb_idx,lb_idb


def step2_2(A,b,H,c,ub_idx,ub_idb,lb_idx,lb_idb):
    # Step 2.2 defines equality constraints in H for variables with zero range 
    # the variable range is provided from step 2.1
    rows_to_delete = []

    if len(lb_idx)>0 and len(ub_idx)>0:
        n_variables = A.shape[1]
        var_bounds = np.concatenate([[np.zeros(n_variables)-np.inf],
                                        [np.zeros(n_variables)+np.inf]],axis=0)
        var_bounds[0,lb_idx] = b[lb_idb]
        var_bounds[1,ub_idx] = b[ub_idb]
        var_bounds = var_bounds.T
        var_ranges = np.diff(var_bounds)

        for var_idx in range(len(var_bounds)):
            # If a variable has zero range 
            if var_ranges[var_idx] <= 0 :
                # Add row to H
                H_new_row = np.zeros(n_variables)
                H_new_row[var_idx] = 1
                c_new_row = var_bounds[var_idx,0]
                H.append(H_new_row)
                c.append(c_new_row)

                # Delete corosponding two rows from A and b
                rows_to_delete.append(lb_idb[np.where(lb_idx==var_idx)][0])
                rows_to_delete.append(ub_idb[np.where(ub_idx==var_idx)][0])

    delete_rows_csr(A,rows_to_delete[::-1])
    b = np.delete(b,rows_to_delete)
    H = np.array(H)
    c = np.array(c)
    print('step 2 done ')
    return A,b,H,c


def step2(A,b,H,c):
    # combining step 2.1 and 2.2
    ub_idx,ub_idb,lb_idx,lb_idb = step2_1(A)
    A,b,H,c = step2_2(A,b,H,c,ub_idx,ub_idb,lb_idx,lb_idb)
    return A,b,H,c


def step3(A,b,H,c):
    # Step 3
    # Compute null space of H and remove all equalities 
    if len(H)>0:
        N = scipy.sparse.csr_matrix(scipy.linalg.null_space(H))
        # find x0 
        res = scipy.optimize.linprog(c=np.zeros(A.shape[1]),A_ub=A,b_ub=b,A_eq=H,b_eq=c)
        x_0 = res.x

        A_new = A.dot(N)
        b_new = b - A.dot(x_0)
        print('reduced model has {} variables'.format(A_new.shape[1]))
    else :
        A_new = A
        b_new = b
        N = None 
        x_0 = None
    print('step 3 done ')
    return A_new,b_new,N,x_0

def delete_rows_csr(mat, i_list):
    # Function for deleting row from matrix on compressed sparse row format
    for i in i_list:
        if not isinstance(mat, scipy.sparse.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")
        n = mat.indptr[i+1] - mat.indptr[i]
        if n > 0:
            mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
            mat.data = mat.data[:-n]
            mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
            mat.indices = mat.indices[:-n]
        mat.indptr[i:-1] = mat.indptr[i+1:]
        mat.indptr[i:] -= n
        mat.indptr = mat.indptr[:-1]
        mat._shape = (mat._shape[0]-1, mat._shape[1])


def decrush(z_samples,N,x_0):
    # Converting samples from Z space to X space
    x_samples = np.array([N.dot(z)+x_0 for z in z_samples])
    return x_samples


def tjek_sample(x,A,sense,b):
    sample_verdict = []
    for lhs,sns,b_i in zip(np.dot(A,x),sense,b):
        if sns == '>':
            sample_verdict.append(lhs>=b_i)
        if sns == '<':
            sample_verdict.append(lhs<=b_i)
        if sns == '=':
            sample_verdict.append(abs(lhs-b_i)<1e-6 )
    if not all(sample_verdict):
        print('err')
    else :
        print('sample ok')

# %% Load model from .lp file
m = gp.read('model_small.lp')
m.printStats()

# Load variable mapping 
with open('model_small_vars.pickle', 'rb') as handle:
    symbol_cuid_pairs = pickle.load(handle)

#%% Add variable bounds ass constraints 

for var in m.getVars():
    if var.getAttr('ub') <np.inf :
        m.addConstr(var,'<',var.getAttr('ub'))
    if var.getAttr('lb')>-np.inf:
        m.addConstr(var,'>',var.getAttr('lb'))
    if var.getAttr('ub')-var.getAttr('lb') == 0:
        print('zero range')

# %% Retriving A matrix, b vector and constraint sense


A_spar = m.getA()
#A = A_spar.toarray()
b = m.getAttr('rhs')
sense = [con.sense for con in m.getConstrs()]

print('model has {} variables'.format(A_spar.shape[1]))
# %% Presolve model

%time A_new,b_new,N,x_0 = presolve(A_spar,b,sense)

#%% Finding intial solution to z problem

m_reduced = gp.Model("matrix1")

x = m_reduced.addMVar(shape=A_new.shape[1], name="x")

obj = np.zeros(A_new.shape[1])
m_reduced.setObjective(obj @ x, GRB.MAXIMIZE)

m_reduced.addConstr(A_new @ x <= b_new, name="c")
m_reduced.update()
m_reduced.optimize()
z_0 = x.X

#%%
n_samples = 1000
z_samples = rand_walk_sample(A=A_new,b=b_new,x_0=z_0,n=n_samples)
x_samples = decrush(z_samples,N,x_0)

# %% Creating dict of data frames to contain data 
symbol_cuid_pairs['ONE_VAR_CONSTANT'] = 'constant[constant]'
variables_symbolic_names = [var.getAttr('VarName') for var in m.getVars()]#[:-1]
variable_names = [str(symbol_cuid_pairs[v]) for v in variables_symbolic_names ]

df_names = [column.split('[')[0] for column in variable_names]
df_columns = [column.split('[')[1][:-1] for column in variable_names]

dataFrames = {}
for df_name in np.unique(df_names):
    filt = [name == df_name for name in df_names]
    columns = np.array(df_columns)[filt]
    dataFrames[df_name] = pd.DataFrame(columns=columns,
                                data=x_samples[:,filt])


#%%
with open('dataFrames.pickle', 'wb') as handle:
    pickle.dump(dataFrames, handle, protocol=pickle.HIGHEST_PROTOCOL)  
print('saved dataFrames as pickle')


#%%
#with open('dataFrames.pickle', 'rb') as handle:
#    test = pickle.load(handle)
# %%


msk = np.random.rand(dataFrames['generator_p_nom'].shape[0])>0.5

#%%
if False :
    stat = stats.ks_2samp(np.array(dataFrames['generator_p_nom'].loc[msk,'ocgt0']),
                        np.array(dataFrames['generator_p_nom'].loc[~msk,'ocgt0']))

    stat2 = stats.ks_2samp(np.array(dataFrames['generator_p_nom'].loc[msk,'wind0']),
                        np.array(dataFrames['generator_p_nom'].loc[~msk,'wind0']))

    print(stat.pvalue,stat2.pvalue)


#%%
if False : 
    fig = go.Figure()
    for gen in dataFrames['generator_p_nom'].keys()[0:10]:
        fig.add_trace(go.Histogram(x=dataFrames['generator_p_nom'].loc[:,gen],  
                                name = gen,
                                xbins=dict( # bins used for histogram
                                #start=0,
                                #end=10,
                                #size=0.1
                            ),))
    fig.update_layout(barmode='overlay') 
    fig.show()                        
    #%%
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dataFrames['generator_p_nom'].loc[msk,'AT ocgt'],  
                                name = 'My gen 0',
                                xbins=dict( # bins used for histogram
                                #start=0,
                                #end=10,
                                size=0.1
                            ),))
    fig.add_trace(go.Histogram(x=dataFrames['generator_p_nom'].loc[msk,'BA ocgt'],
                                name = 'My gen 1',
                                xbins=dict( # bins used for histogram
                                                            start=0,
                                                            #end=10,
                                                            #size=0.1
                                                        ),))
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dataFrames['generator_p_nom'].loc[~msk,'AT ocgt'],  
                                name = 'My gen 0',
                                xbins=dict( # bins used for histogram
                                start=0,
                                end=10,
                                size=0.1
                            ),))
    fig.add_trace(go.Histogram(x=dataFrames['generator_p_nom'].loc[~msk,'BA ocgt'],
                                name = 'My gen 1',
                                xbins=dict( # bins used for histogram
                                                            start=0,
                                                            end=10,
                                                            size=0.1
                                                        ),))
    fig.show()


    # %%
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dataFrames['passive_branch_s_nom'].loc[:,'Line,My line 0'],  
                                name = 'My line 0',
                                xbins=dict( # bins used for histogram
                                start=0,
                                end=10,
                                size=0.5
                            ),))
    fig.add_trace(go.Histogram(x=dataFrames['passive_branch_s_nom'].loc[:,'Line,My line 1'],  
                                name = 'My line 1',
                                xbins=dict( # bins used for histogram
                                start=0,
                                end=10,
                                size=0.5
                            ),))
    #fig.add_trace(go.Histogram(x=dataFrames['passive_branch_s_nom'].loc[:,'Line,My line 2'],  
    #                            name = 'My line 2',
    #                            xbins=dict( # bins used for histogram
    #                            start=0,
    #                            end=10,
    #                            size=0.5
    #                        ),))                        



#%%

row = np.array([0, 0, 1, 2, 2, 2,3,4])

col = np.array([0, 2, 2, 0, 1, 2,1,2])

data = np.array([1, 2, 1, 4, 5, 6, 1,1])

A_spar = scipy.sparse.csr_matrix((data, (row, col)), shape=(6, 3))

A = A_spar.toarray()
b = [0,0,3,1,0,0]
sense = ['>','>','<','=','<','>']


# %%

A_new,b_new,N,x_0 = presolve_new(A_spar,b,sense)

#%%


A_new,b_new,N,x_0 = presolve(A,b,sense)

# %%


# %%

def presolve_old(A,b,sense):
    # Step 1
    # Remove any unececary constraints and flip signs to correct form 
    # Split A into A_new and H_new such A_new contains < constraints and H contains =
    #A_new = np.empty([0,A.shape[1]])
    #b_new = np.empty(0)
    #H_new = np.empty([0,A.shape[1]])
    #c_new = np.empty(0)
    A_new = []
    b_new = []
    H_new = []
    c_new = []
    for i in range(len(b)):
        if all(A[i,:]==0):
            pass
        elif sense[i] == '>':
            A_new.append(-A[i,:])
            b_new.append(-b[i])
        elif sense[i] == '<':
            A_new.append(A[i,:])
            b_new.append(b[i])
        elif sense[i] == '=':
            H_new.append(A[i,:])
            c_new.append(b[i])
    print('step 1 done ')
    
    # Step 2.1 
    # Calculate variabl bounds

    ub_idx = []
    ub_idb = []
    lb_idx = []
    lb_idb = []
    for i,row in enumerate(A_new):
        filt = list(row!=0)
        if sum(filt)==1 :
            if row[filt][0]>0:
                ub_idx.append(filt.index(True))
                ub_idb.append(i)
            elif row[filt][0]<0:
                lb_idx.append(filt.index(True))
                lb_idb.append(i)

    ub_idx = np.array(ub_idx)
    ub_idb = np.array(ub_idb)
    lb_idx = np.array(lb_idx)
    lb_idb = np.array(lb_idb)
    A_new = np.array(A_new)
    b_new = np.array(b_new)
    print('step 2.1 done ')

    # Step 2.2 
    # Add equalities when variables have zero range 
    if len(lb_idx)>0 and len(ub_idx)>0:
        n_variables = len(A_new[0])
        var_bounds = np.concatenate([[np.zeros(n_variables)-np.inf],
                                        [np.zeros(n_variables)+np.inf]],axis=0)
        var_bounds[0,lb_idx] = b_new[lb_idb]
        var_bounds[1,ub_idx] = b_new[ub_idb]
        var_bounds = var_bounds.T

        var_ranges = np.diff(var_bounds)

        for var_idx in range(len(var_bounds)):
            if var_ranges[var_idx] <= 0 :
                H_new_row = np.zeros(A_new.shape[1])
                H_new_row[var_idx] = 1
                c_new_row = var_bounds[var_idx,0]
                H_new.append(H_new_row)
                c_new.append(c_new_row)

                #del A_new[]
                A_new = np.delete(A_new,lb_idb[np.where(lb_idx==var_idx)[0]],0)
                b_new = np.delete(b_new,lb_idb[np.where(lb_idx==var_idx)[0]],0)
                A_new = np.delete(A_new,ub_idb[np.where(ub_idx==var_idx)[0]],0)
                b_new = np.delete(b_new,ub_idb[np.where(ub_idx==var_idx)[0]],0)
    
    H_new = np.array(H_new)
    c_new = np.array(c_new)
    print('step 2 done ')
    
    #A_new = np.array(A_new)
    #b_new = np.array(b_new)
    #H_new = np.array(H_new)
    #c_new = np.array(c_new)
    # Step 3
    # Compute null space of H and remove all equalities 
    if len(H_new)>0:
        N = scipy.linalg.null_space(H_new)
        # find x0 
        res = scipy.optimize.linprog(c=np.zeros(A_new.shape[1]),A_ub=A_new,b_ub=b_new,A_eq=H_new,b_eq=c_new)
        x_0 = res.x
        # x0 tjek
        if not all(np.dot(H_new,x_0) - c_new < 1e-3):
            print('Equalities violated')
        elif not all(np.dot(A_new,x_0) <= b_new):
            print('Inequalities violotaed')


        A_new2 = np.dot(A_new,N)
        b_new2 = b_new - np.dot(A_new,x_0)
        print('reduced model has {} variables'.format(A_new2.shape[1]))
    else :
        A_new2 = A_new
        b_new2 = b_new
        N = None 
        x_0 = None
    print('step 3 done ')
    return A_new2,b_new2,N,x_0