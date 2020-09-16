#%%
import time
import numpy as np
import dask.array as da 
import logging

def sample(A,b,x_0,n,time_max=1):
    timer = time.time()
    timer_total = time.time()
    #

    A_dot_x0= A.dot(x_0)
    x_samples = [x_0]
    #for i in range(n):
    while len(x_samples)<n  :
        # reconsider this section
        # red this https://towardsdatascience.com/the-best-way-to-pick-a-unit-vector-7bd0cc54f9b
        
        # Draw random direction theta
        f = np.random.rand(A.shape[1])*2-1
        theta = f/np.cos(f)
        #theta = f 
        # define available lengths t_possible before violating constraints
        A_dot_theta = A.dot(theta)
        t_possible = (b-A_dot_x0)/A_dot_theta
        # Find maximum and minimum allowable step size in positive theta and negetive theta direction
        t_max = t_possible[A_dot_theta>1e-6].min()
        t_min = t_possible[A_dot_theta<-1e-6].max()
        t_len = t_max-t_min
        # if max step length is very short, i.e. standing in corner, then draw a new theta
        if t_len < 1e-12:
            #print('t less than 1e-12, t = {}'.format(t_len))
            if time.time()-timer>time_max:
                logging.warning('did not manage to find viable sample within specified time')
                break 
            else :
                continue 
        else : 
            timer = time.time()
        # draw random step size 
        t = np.random.rand()*t_len+t_min
        # Define new sample point 
        x_new = x_0 + theta*t
        A_dot_x_new = A.dot(x_new)
        # If new sample point is not violating constraints, add to list 
        if all((A_dot_x_new-b)<1e-6):
            x_samples.append(x_new)
            x_0 = x_new
            A_dot_x0 = A_dot_x_new
        else :
            print('discarding sample')
    #x_samples = np.array(x_samples)
    x_samples = da.from_array(x_samples,chunks='auto')
    #print('elapsed time {}'.format(time.time()-timer_total))
    return x_samples


#%%
if __name__ == '__main__':
    import plotly.graph_objects as go

    A = np.array([[1,0],
                [0,1],
                [-1,0],
                [0,-1]])
    b = np.array([1,1,0,0])

    x_0 = np.array([0.0,0.])
    n = 10

    samples = sample(A,b,x_0,n)


    fig = go.Figure(go.Scatter(x=samples[:,0],y=samples[:,1],mode='markers+lines'))
    fig.show()

# %%
