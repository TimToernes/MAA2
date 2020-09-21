#%%
import numpy as np
import plotly.graph_objects as go

#%%

A = np.array([[0,-1],
              [-1,0],
              [1,1]])
b = np.array([0,0,1])
x_0 = [0,0.5]


#%%

A = np.array([[1,1],
              [1,-1],
              [-1,-1],
              [-1,1]])
b = np.array([1,0.5,1,0.5])

x_0 = [0,0]

# %%

fig = go.Figure(go.Scatter(x=[1,0],y=[0,1]))
fig.add_trace(go.Scatter(x=[0,-1],y=[1,0]))
fig.add_trace(go.Scatter(x=[-1,0],y=[0,-1]))
fig.add_trace(go.Scatter(x=[0,1],y=[-1,0]))
# %%

n_samples = 10000
x_samples = [] 
for i in range(n_samples):
    A_dot_x0 = np.dot(A,x_0)
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
    # draw random step size 
    t = np.random.rand()*t_len+t_min
    # Define new sample point 
    x_new = x_0 + theta*t
    A_dot_x_new = A.dot(x_new)
    A_dot_x0 = A_dot_x_new
    x_0 = x_new
    x_samples.append(x_new)
x_samples = np.array(x_samples)
# %%
fig.add_trace(go.Scatter(x=x_samples[:,0],y=x_samples[:,1],mode='markers'))
# %%

n_samples = 1000
A = A[:,0]
x_0 = x_0[0]

#%%
x_samples = [] 
for i in range(n_samples):
    #A_dot_x0 = np.dot(A,x_0)
    A_dot_x0 = A*x_0
    f = np.random.rand(1)*2-1
    #f = np.concatenate([f,np.zeros(1)])
    theta = f/np.cos(f)
    #theta = f 
    # define available lengths t_possible before violating constraints
    #A_dot_theta = A.dot(theta)
    A_dot_theta = A*theta
    t_possible = (b-A_dot_x0)/A_dot_theta
    # Find maximum and minimum allowable step size in positive theta and negetive theta direction
    t_max = t_possible[A_dot_theta>1e-6].min()
    t_min = t_possible[A_dot_theta<-1e-6].max()
    t_len = t_max-t_min
    # if max step length is very short, i.e. standing in corner, then draw a new theta
    # draw random step size 
    t = np.random.rand()*t_len+t_min
    # Define new sample point 
    x_new = x_0 + theta*t
    #A_dot_x_new = A.dot(x_new)
    A_dot_x_new = A*x_new
    A_dot_x0 = A_dot_x_new
    x_0 = x_new
    x_samples.append(x_new)
x_samples = np.array(x_samples)
# %%

fig.add_trace(go.Scatter(x=x_samples[:,0],y=np.zeros(len(x_samples)),mode='markers'))
fig.show()
# %%

# %%


n = 3  # dimension of the original polytope
p = 1   # dimension of the projected polytope

# Original polytope:
# - inequality constraints: \forall i, |x_i| <= 1
# - equality constraint: sum_i x_i = 0
#A = vstack([+eye(n), -eye(n)])
#b = ones(2 * n)
C = ones(n).reshape((1, n))
d = array([0])
ineq = (A, b)  # A * x <= b
eq = (C, d)    # C * x == d

# Projection is proj(x) = [x_0 x_1]
E = zeros((p, n))
E[0, 0] = 1.
#E[1, 1] = 1.
f = zeros(p)
proj = (E, f)  # proj(x) = E * x + f

vertices = project_polytope(proj, ineq, eq, method='bretl')
# %%

n = 20  # dimension of the original polytope
p = 2   # dimension of the projected polytope


A = np.concatenate([-np.diag(np.ones(n)),np.diag(np.ones(n))])
b = np.concatenate([np.zeros(n),np.ones(n)])


ineq = (A, b)  # A * x <= b

E = zeros((p, n))
E[0, 0] = 1.
E[1, 1] = 1.

f = zeros(p)
proj = (E, f)  # proj(x) = E * x + f

vertices = project_polytope(proj, ineq)
vertices
# %%
