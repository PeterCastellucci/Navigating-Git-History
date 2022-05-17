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
