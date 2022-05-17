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