#画图部分可以直接用来画图
import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling，用于拉丁超立方抽样，虽然当前代码中未实际用到，但可能后续会扩展功能用到该模块
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
def _draw_contour_and_surface(T, X, F_xt):  
    # 创建一个包含一个子图的图形对象和对应的坐标轴对象
    fig, ax = plt.subplots(1, 1)  
    # 在当前坐标轴上绘制等高线图，参数20表示绘制20条等高线，cmap="rainbow"指定使用彩虹颜色映射
    cp = ax.contour(T, X, F_xt, 20, cmap="rainbow")  
    # 为等高线图添加颜色条，方便查看数值对应的颜色范围
    fig.colorbar(cp)  
    # 设置子图的标题为'F(x, t)'，用于标识所绘制的函数
    ax.set_title('F(x, t)')  
    # 设置x轴的标签为't'，表明该轴对应的变量
    ax.set_xlabel('t')  
    # 设置y轴的标签为'x'，表明该轴对应的变量
    ax.set_ylabel('x')  
    # 显示绘制好的二维等高线图
    plt.show()  
    # 创建一个三维坐标轴对象，用于后续绘制三维曲面图
    ax = plt.axes(projection='3d')  
    # 在三维坐标轴上绘制曲面图，传入的参数需要转换为numpy数组格式，这里展示函数值随x和t变化的三维曲面情况
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")  
    # 设置x轴的标签为't'    ax.set_xlabel('t')  
    ax.set_ylabel('x')  
    # 设置z轴的标签为'f(x,t)'，表明该轴对应的是函数值
    ax.set_zlabel('f(x,t)')  
    # 显示绘制好的三维曲面图
    plt.show()

# 图像绘制函数，用于对输入的x、t以及对应的函数值y进行处理后绘制图形
# 该函数会先对输入的x和t的维度进行合理性检查和处理，再调用_draw_contour_and_surface函数进行实际绘图
def plot3D(x, t, y):  
    # 检查x的维度，如果是二维且第二维的大小为1（即形状类似(n, 1)），则去除第二维
    if x.dim() == 2 and x.shape[1] == 1:  
        x_plot = x.squeeze(1)
    # 如果x本身就是一维张量，则直接使用
    elif x.dim() == 1:  
        x_plot = x
    # 如果维度不符合预期，则抛出异常提示输入的x维度有问题
    else:  
        raise ValueError("Input x has unexpected dimensions")
    # 对t做类似x的维度检查和处理
    if t.dim() == 2 and t.shape[1] == 1:  
        t_plot = t.squeeze(1)
    elif t.dim() == 1:
        t_plot = t
    else:
        raise ValueError("Input t has unexpected dimensions")
    # 使用处理后的x_plot和t_plot生成二维坐标网格，明确指定索引方式为'ij'，符合未来版本要求且避免歧义
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')  
    # 获取对应的函数值，用于后续绘图
    F_xt = y  
    # 调用公共的绘图函数来绘制等高线图和三维曲面图
    _draw_contour_and_surface(T, X, F_xt)

# 图像绘制函数，直接使用传入的x、t以及对应的函数值y进行绘图
# 与plot3D函数不同的是，它不做额外的维度处理，直接将传入的坐标用于绘图
def plot3D_Matrix(x, t, y):  
    X, T = x, t
    F_xt = y
    _draw_contour_and_surface(T, X, F_xt)
    
'''
def f_real(x, t):  
    return torch.exp(-t) * (torch.sin(np.pi * x))
x = torch.linspace(-1, 1, 200).view(-1, 1)  
t = torch.linspace(0, 1, 100).view(-1, 1)  
# 使用经过处理（去除多余维度）的x和t生成二维坐标网格，为后续计算函数值和绘图做准备
X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')  
y_real = f_real(X, T)  
# 调用plot3D函数，传入生成的数据进行可视化展示，绘制出二维等高线图和三维曲面图
plot3D(x, t, y_real)'''