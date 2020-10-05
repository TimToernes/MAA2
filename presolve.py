#%%
import scipy 
import scipy.sparse
from scipy import stats
import scipy.optimize
import numpy as np
import pandas as pd
import dask.array as da
#import sparseqr
import logging
logger = logging.getLogger()


def presolve(A,b,sense,m):
    # A     : sparse n*d array describing optimization problem 
    # b     : n*1 array
    # sense : n*1 list containing sense on the form: =,<,>
    # m     : Gurobi model of the problem 
    # 
    # Given a problem containing a mix of equalities and inequalities,
    # the presolve function will find a fully dimensional sub problem. 
    # Meaning that it returns a full rank A matrix with the belonging b vector
    # 

    logger.info('Presolve started')
    A,b,H,c = _step1(A,b,sense)
    A,b,H,c = _step2(A,b,H,c)
    A,b,N,x_0 = _step3(A,b,H,c,m)
    logger.info('Presolve done') 

    return A,b,N,x_0


def _step1(A,b,sense):
    # Step 1 - Sorting of the raw A array 
    # Empty rows are removed
    # > constraints are fliped to <
    # = constraints are moved from the A to the H array
    b_1 = []
    H_1 = scipy.sparse.csr_matrix((0,A.shape[1]))
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
            #H_1.append(A.getrow(id_row).toarray()[0])
            H_1 = scipy.sparse.vstack([H_1,A.getrow(id_row)])
            c_1.append(b[id_row])
            rows_to_delete.append(id_row)

    _delete_rows_csr(A,rows_to_delete[::-1])
    b_1 = np.array(b_1) 
    logger.info('step 1 done')
    return A,b_1,H_1,c_1

def _step2(A,b,H,c):
    ub_idx,ub_idb,lb_idx,lb_idb = _step2_1(A)
    A,b,H,c = _step2_2(A,b,H,c,ub_idx,ub_idb,lb_idx,lb_idb)
    return A,b,H,c

def _step2_1(A):
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
    logger.info('step 2.1 done')
    return ub_idx,ub_idb,lb_idx,lb_idb

def _step2_2(A,b,H,c,ub_idx,ub_idb,lb_idx,lb_idb):
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
                H = scipy.sparse.vstack([H,H_new_row])
                #H.append(H_new_row)
                c.append(c_new_row)

                # Delete corosponding two rows from A and b
                rows_to_delete.append(lb_idb[np.where(lb_idx==var_idx)][0])
                rows_to_delete.append(ub_idb[np.where(ub_idx==var_idx)][0])

    _delete_rows_csr(A,rows_to_delete[::-1])
    b = np.delete(b,rows_to_delete)
    c = np.array(c)
    logger.info('step 2 done ')
    return A,b,H,c


def _step3(A,b,H,c,m):
    # Step 3
    # Compute null space of H and remove all equalities 
    if H.shape[0]>0:
        #N = scipy.sparse.csr_matrix(scipy.linalg.null_space(H.toarray()))
        N = _calc_null_space(H,tjek=True)
        # find x0 
        #res = scipy.optimize.linprog(c=np.zeros(A.shape[1]),A_ub=A,b_ub=b,A_eq=H,b_eq=c)
        #x_0 = res.x
        #x_0 = find_feasible_solution(A,b,H,c)
        m.optimize()
        x_0 = m.X

        A_new = A.dot(N)
        b_new = b - A.dot(x_0)
        logger.info('reduced model has {} variables'.format(A_new.shape[1]))
    else :
        A_new = A
        b_new = b
        N = None 
        x_0 = None
    logger.info('step 3 done ')
    return A_new,b_new,N,x_0

def _calc_null_space(A_spar,tjek=False):
    try :
        import sparseqr
    except :
        logger.warning('Did not finde sparseqr. Using numpy')
        Q,_ = np.linalg.qr(A_spar.transpose().toarray())
        r = np.linalg.matrix_rank(A_spar.transpose().toarray(),tol=1)
        N = Q[:,r:]
        N = scipy.sparse.csr_matrix(N)
    else : 
        Q, _, _,r = sparseqr.qr( A_spar.transpose() )
        del _ 
        N = Q.tocsr()[:,r:]
    if tjek :
        if A_spar.dot(N).max()>1e-3:
            logger.warning('Nullspace tollerence violated')
        else :
            logger.info('Nullspace is good')
    return N


def _delete_rows_csr(mat, i_list):
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
    #x_samples = np.array([N.dot(z)+x_0 for z in z_samples])
    x_samples = da.from_array([N.dot(z)+x_0 for z in z_samples],chunks='auto')
    return x_samples


#%% Test section
if __name__ == '__main__':
    import plotly.graph_objects as go
    import plotly.express as px
    from create_test_model import create_model
    import gurobipy as gp
    import os 

    name = 'models/small_model'
    if not os.path.isfile(name+'.lp'):
        create_model(n_buses = 3,
                n_snapshots = 30,
                name=name,
                p_min_PV = 0,
                co2_constraint=None)

    m = gp.read(name+'.lp')
    m.printStats()

    A_spar = m.getA()
    b = m.getAttr('rhs')
    sense = [con.sense for con in m.getConstrs()]

    A,b,N,x_0 = presolve(A_spar,b,sense,m)

    if np.linalg.matrix_rank(A.toarray())- min(A.shape) == 0 :
        print('test succesfull')


# %%
