#%%
import numpy as np
import sympy
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import scipy.linalg
import plotly.io as pio
pio.renderers.default = "browser"
# %%

#A = np.array([[-1,0],[0,-1],[1,1],[1,0],[0,1]])
#C = np.array([0,0,1,0.8,0.7])

#A = np.array([[-1,0,0],[0,-1,0],[0,0,-1],[1,1,1]])
#C = np.array([0,0,0,1,])

dim = 10000

A = np.diag(-np.ones(dim))
A = np.concatenate([A,np.diag(np.ones(dim))],axis=0)


b = np.concatenate([np.zeros(dim),np.ones(dim)])

H = np.zeros([int(np.floor(dim/2)),dim])
for i in range(int(np.floor(dim/2))):
    for j in range(dim):
        if j%2 == 0:
            H[i,j+2*i] = 1
            H[i,j+1+2*i] = 1
            break

c = np.ones(int(np.floor(dim/2))).T




# %%

def sample(A,b,x_0,n):
    timer = time.time()
    timer_total = time.time()
    #

    x_samples = []
    for i in range(n):
        # Draw random direction
        f = np.random.rand(A.shape[1])*2-1
        theta = f/np.cos(f)
        # define available lengths in given directions 
        A_dot_x0 = np.dot(A,x_0)
        dirs = (b-A_dot_x0)/np.dot(A,theta)
        # Remove lengths violating constraints 
        dirs = dirs[b-A_dot_x0>=0]
        # draw random step size 
        try :
            t_max = dirs[dirs>0].min()
        except : 
            t_max = 0
        try :
            t_min = dirs[dirs<0].max()
        except : 
            t_min = 0 
        t_len = t_max-t_min
        t = np.random.rand()*t_len+t_min
        # Define new sample point 
        x_new = x_0 + theta*t
        x_0 = x_new
        # If new sample point is not violating constraints, add to list 
        if not any((b-A_dot_x0)<0):
            x_samples.append(x_new)
    x_samples = np.array(x_samples)
    print('elapsed time {}'.format(time.time()-timer_total))
    return x_samples

# %%
x_0 = np.array([0.5,0.5,0.5])
#x_0 = np.array([0,1,0.5])
# x0 tjek
print(np.dot(H,x_0) == c)
print(np.dot(A,x_0) <= b)


x_samples = sample(A,b,x_0,1000)

#msk = np.random.rand(len(x_samples))>0.3

msk = np.random.rand(len(x_samples))>0.1

fig = go.Figure(data=[go.Scatter3d(x=x_samples[msk,0], y=x_samples[msk,1], z=x_samples[msk,2],
                                   mode='markers')])
fig.show()


# %%

def licols(A,toll=1e-10):
# https://www.mathworks.com/matlabcentral/answers/104739-integer-constrained-optimization-using-the-ga-genetic-algorithm-solver-of-matlab-can-anyone-he#answer_114013
#
#Extract a linearly independent set of columns of a given matrix X
#
#    [Xsub,idx]=licols(X)
#
#in:
#
#  X: The given input matrix
#  tol: A rank estimation tolerance. Default=1e-10
#
#out:
#
# Xsub: The extracted columns of X
# idx:  The indices (into X) of the extracted columns



    Q,R,P = scipy.linalg.qr(A,pivoting=True)

    #if ~isvector(R)
    diagr = np.abs(np.diag(R))
    #else
    #diagr = R(1)

    #Rank estimation
    #r = find(diagr >= tol*diagr(1), 1, 'last') #rank estimation
    r = sum(diagr >= toll*diagr[0])

    idx2 = np.sort(P[:r])
    A2 = A[:,idx2]
    idx1 = np.sort(P[r:])
    A1 = A[:,idx1]

    return A1,A2,idx1,idx2
#%% Split problem method 

A1,A2,idx1,idx2 = licols(H)

x1 = np.array([0,1])

x2 = np.linalg.lstsq(A2,c)[0] - np.dot(np.linalg.lstsq(A2,A1)[0],x1)



#%% Null space method 


F = scipy.linalg.null_space(H)

x_0 = np.zeros(dim)+0.5
# x0 tjek
print(np.dot(H,x_0) == c)
print(np.dot(A,x_0) <= b)

A_new = np.dot(A,F)
b_new = b - np.dot(A,x_0)

#z_0 = np.array([0,0,0])
z_0 = np.zeros(A_new.shape[1])

print(np.dot(A_new,z_0) <= b_new)

# %%

z_samples = sample(A_new,b_new,z_0,1000)
x_samples = np.array([np.dot(F,z)+x_0 for z in z_samples])

msk = np.random.rand(len(x_samples))>0
# %%


fig = go.Figure(data=[go.Scatter3d(x=x_samples[msk,0], y=x_samples[msk,1], z=x_samples[msk,2],
                                   mode='markers')])
fig.show()


# %%

n_plots = 10

from plotly.subplots import make_subplots
fig = make_subplots(rows=n_plots, cols=1)

for i in range(n_plots):
    fig.append_trace(go.Histogram(x=x_samples[msk,i+10]),row=i+1,col=1)
fig.show()

# %%
