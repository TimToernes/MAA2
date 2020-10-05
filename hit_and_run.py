#%%
import time
import numpy as np
import dask.array as da 
import logging
logger = logging.getLogger()




def sampler(A,b,x_0,n,time_max=1):
    timer = time.time()
    timer_total = time.time()
    #
    A_dot_x0= A.dot(x_0)
    x_samples = [x_0]

    while len(x_samples)<n  :
        
        if (time.time()-timer)>time_max:
            logger.warning('did not manage to find viable sample within specified time')
            break 
        # Draw random direction theta
        theta = draw_random_dir(A.shape[1])

        # define available lengths lambdas before violating constraints
        lambdas,A_dot_theta = calc_lambdas(A,b,theta,x_0)

        if sum(lambdas>0)<1:
            lambda_max = 0
        else :
            lambda_max = np.nanmin(lambdas[lambdas>1e-6])
        if sum(lambdas<=0)<1:
            lambda_min = 0
        else :
            lambda_min = np.nanmax(lambdas[lambdas<1e-6])

        # if max step length is very short, i.e. standing in corner, then draw a new theta
        if lambda_max-lambda_min <= 0 :
            logger.info('Not possible to move with current direction')
            continue
 
        # draw random step size 
        lam = np.random.uniform(low=lambda_min, high=lambda_max)
        # Define new sample point 
        x_new = x_0 + theta*lam
        A_dot_x_new = A.dot(x_new)
        # If new sample point is not violating constraints, add to list 
        if all((A_dot_x_new-b)<1e-1):
            x_samples.append(x_new)
            x_0 = x_new
            A_dot_x0 = A_dot_x_new
            timer = time.time()
        else :
            logger.info('discarding sample - violating constraints')
        
    logger.info('{} samples taken'.format(len(x_samples)))
    #x_samples = np.array(x_samples)
    x_samples = da.from_array(x_samples,chunks='auto')
    #print('elapsed time {}'.format(time.time()-timer_total))
    return x_samples

def calc_lambdas(A,b,theta,x_0):
    A_dot_x0= A.dot(x_0)
    A_dot_theta = A.dot(theta)
    lambdas = (b-A_dot_x0)/A_dot_theta
    lambdas[abs(A_dot_theta)<1e-6] = np.nan
    # Find maximum and minimum allowable step size in positive theta and negetive theta direction
    #lambda_max = lambdas[lambdas>1e-6].min()
    #lambda_min = lambdas[lambdas<-1e-6].max()
    return lambdas, A_dot_theta

def draw_random_dir(dim):
    # Draws a random point on the unit sphere of dim dimentions
    #
    # reconsider this section
    # read this https://towardsdatascience.com/the-best-way-to-pick-a-unit-vector-7bd0cc54f9b
    f = np.random.rand(dim)*2-1
    theta = f/np.cos(f)
    theta = theta/np.linalg.norm(theta)
    return theta

def _n_dim_unit_cube(dim):

    A = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))])
    A_spar = scipy.sparse.csr_matrix(A)
    b = np.concatenate([np.ones(dim),np.zeros(dim)]) 
    x_0 = np.zeros(dim)+0.1

    return A_spar,b,x_0

#%%
if __name__ == '__main__':
    import plotly.graph_objects as go
    import scipy.sparse

    dim = 10
    A_spar,b,x_0 = _n_dim_unit_cube(dim)

    n = 1000
    samples = sampler(A_spar,b,x_0,n,time_max=10)

    fig = go.Figure(go.Scatter(x=samples[:,0],y=samples[:,1],mode='markers'))
    fig.show()

    std_unifor_dist = 1/np.sqrt(12)

    print('std error',abs(np.std(samples[:,0].compute()) - std_unifor_dist))
    print('mean error',abs(np.mean(samples[:,0].compute())-0.5))
    

# %%
