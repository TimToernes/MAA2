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


# %%

def presolve(A,b,sense):
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

def decrush(z_samples,N,x_0):
    x_samples = np.array([np.dot(N,z)+x_0 for z in z_samples])
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
m = gp.read('test.lp')
m.printStats()

# Load variable mapping 
with open('var_pairs.pickle', 'rb') as handle:
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


A = m.getA().toarray()
b = m.getAttr('rhs')
sense = [con.sense for con in m.getConstrs()]

print('model has {} variables'.format(A.shape[1]))
# %% Presolve model

A_new,b_new,N,x_0 = presolve(A,b,sense)


#%%
m_reduced = gp.Model("matrix1")

x = m_reduced.addMVar(shape=A_new.shape[1], name="x")

obj = np.zeros(A_new.shape[1])
m_reduced.setObjective(obj @ x, GRB.MAXIMIZE)

m_reduced.addConstr(A_new @ x <= b_new, name="c")
m_reduced.update()
m_reduced.optimize()
z_0 = x.X

#%%
n_samples = 100000
z_samples = rand_walk_sample(A=A_new,b=b_new,x_0=z_0,n=n_samples)
x_samples = decrush(z_samples,N,x_0)
x_samples_uncut = x_samples.copy()
#x_samples = x_samples[:,:-1] # removing last variable as it is just a constant used for adding constant term to objective 
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



