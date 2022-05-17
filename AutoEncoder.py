# -*- coding: utf-8 -*-
"""
Generate synthetic low dimensional manifold in a 6D space.

Task for students: Extend the script to find the manifold
using an auto-encoder. See assignment details on Blackboard
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
import time

student_id=10050985

np.random.seed(student_id)

n_samples=300

def make_manifold_samples(n_samples):
    """Generate samples in a low dimensional space"""
    X=np.zeros([n_samples,5])
    X[:,0]=np.arange(-2,2,4/n_samples)
    X[:,1]=np.random.random(n_samples)-0.5
    if (student_id%2==0):
        X[:,2]=np.random.random(n_samples)-0.5
    X[:,3]=np.sin(4*X[:,0])
    if (student_id%3==0):
        X[:,4]=X[:,0]**2
    return X

X4=make_manifold_samples(n_samples)
plt.plot(X4[:,0],X4[:,1],"r+")
plt.plot(X4[:,0],X4[:,2],"g+")
plt.plot(X4[:,0],X4[:,3],"y+")
plt.plot(X4[:,0],X4[:,4],"b+")
plt.show()

# Generate random projection
P=np.random.random([6,5])

X6=X4@P.T

print("Shape: ",X6.shape)

Xt=torch.tensor(X6,dtype=torch.float32)

# Number of samples
n_samples=100

# Number of dimensions
nd=6

N_hidden=20  # Number of dimensions of hidden layer
n_latent = 1
# Use the nn package to define a model as a sequence of layers. 
# nn.Sequential is a Module which contains other Modules, 
# and applies them in sequence to produce its output. 
# Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(nd, N_hidden),
    torch.nn.Sigmoid(),
    torch.nn.Linear(N_hidden, n_latent),
    torch.nn.Linear(n_latent, N_hidden),
    torch.nn.Sigmoid(),
    torch.nn.Linear(N_hidden,nd)
)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.uniform_(-1,1)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

loss_fn = torch.nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of samples per batch
# Must divide number of samples exactly
batch_size=25

# Number of complete passes through all data
n_epochs=15000

# Create a random number generator to make permutations
#rng=np.random.default_rng()
