import numpy as np
import time
from pyDOE import lhs
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from sklearn.model_selection import train_test_split
import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def f_real(x, t):
    return torch.exp(-t) * (torch.sin(np.pi * x))

def _draw_contour_and_surface(T, X, F_xt):
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)
    ax.set_title('F(x, t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()

def plot3D(x, t, y):
    if x.dim() == 2 and x.shape[1] == 1:
        x_plot = x.squeeze(1)
    elif x.dim() == 1:
        x_plot = x
    else:
        raise ValueError("Input x has unexpected dimensions")
    if t.dim() == 2 and t.shape[1] == 1:
        t_plot = t.squeeze(1)
    elif t.dim() == 1:
        t_plot = t
    else:
        raise ValueError("Input t has unexpected dimensions")
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    F_xt = y
    _draw_contour_and_surface(T, X, F_xt)

def plot3D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y
    _draw_contour_and_surface(T, X, F_xt)

x = torch.linspace(-1, 1, 200).view(-1, 1)
t = torch.linspace(0, 1, 100).view(-1, 1)
X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')
y_real = f_real(X, T)
plot3D(x, t, y_real)