import torch
from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import math
import time

np.random.seed(1234)

# CUDA
# 判断CUDA可用性
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available")
else:
    device = torch.device('cpu')
    print("Only cpu is available")


# DNN
class DNN(torch.nn.Module):
    def __init__(self, layers,lb,ub):
        self.lb = lb
        self.ub = ub

        super(DNN, self).__init__()

        # 神经网络深度
        self.depth = len(layers) - 1

        # 除了最后一层激活函数都为Tanh
        self.activation = torch.nn.Tanh
        # 用于存储网络层和激活函数的元组列表
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        # 添加最后一个全连接层，即输出层   
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # 构建 Sequential 模型
        self.layers = torch.nn.Sequential(layerDict)

        # 初始化权重 - Xavier初始化部分
        for m in self.layers.modules():
            if isinstance(m, torch.nn.Linear):
                # Xavier正态分布初始化权重
                torch.nn.init.xavier_normal_(m.weight)
                # 偏置初始化为零
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入归一化到[-1, 1]
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        out = self.layers(H)
        return out

#构建和初始化基于物理信息的神经网络
class PhysicsInformedNN:
    """
        初始化 PhysicsInformedNN 类

        参数:
        X_u (numpy.ndarray): 边界条件的数据点，是一个二维数组，每一行代表一个数据点(x,t)
        u (numpy.ndarray): 边界条件对应的目标值
        X_f (numpy.ndarray): 内部数据点，用于满足物理方程
        layers (list): 一个列表，指定神经网络各层的神经元数量
        lb (list or numpy.ndarray): 输入数据的下界
        ub (list or numpy.ndarray): 输入数据的上界
        nu (float): 物理方程中的一个参数，例如粘性系数等
    """
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):

        #边界条件
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        #将其转换为可求导的 PyTorch 张量，同时移动到指定设备上
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)#第一列x
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)#第二列t

        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)#第一列x
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)#第二列t

        self.u = torch.tensor(u).float().to(device)

        self.layers = layers
        self.nu = nu

        # 初始化深度神经网络
        # 使用传入的 layers 列表构建 DNN 实例，并将其移动到指定设备上
        self.dnn = DNN(layers,self.lb,self.ub).to(device)

        # 初始化优化器
        # 使用 LBFGS 优化器来优化神经网络的参数
        # lr: 学习率
        # max_iter: 最大迭代次数
        # max_eval: 最大评估次数
        # tolerance_grad: 梯度的收敛阈值 当梯度的范数小于该值时 优化过程停止
        # tolerance_change: 参数变化的收敛阈值 当参数的变化小于该值时 优化过程停止
        # line_search_fn: 线搜索函数
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-11,
            tolerance_change=1e-11,
            history_size = 100,
            line_search_fn="strong_wolfe"   # 使用强 Wolfe 线搜索方法来确定步长
        )

        self.iter = 0

    def net_u(self, x, t):
        # 将 x 和 t 沿着第 1 个维度（列方向）拼接起来
        # 这样可以将两个特征合并成一个输入张量，以便输入到神经网络中(x,t)--->(x_1,t_1,x_2,t_2,...)
        u = self.dnn(torch.cat([x, t], dim=1)) # 将拼接后的输入张量传入深度神经网络 self.dnn 进行前向传播
        return u
    
    #计算残差
    def net_f(self, x, t):
        
        # 首先调用 net_u 方法计算神经网络对输入 (x, t) 的输出 u
        u = self.net_u(x, t)

        # 计算 u 关于 t 的一阶偏导数 u_t
        # torch.autograd.grad 是 PyTorch 中用于自动求导的函数
        # grad_outputs=torch.ones_like(u) 表示对 u 的每个元素都求梯度
        # retain_graph=True 表示保留计算图，以便后续继续求导
        # create_graph=True 表示创建计算图，使得可以对该导数再次求导

        #u：作为 outputs 参数，表示需要求导的目标张量
        #t：作为 inputs 参数，表示求导的自变量张量
        #[0]：torch.autograd.grad 函数返回的是一个元组，因此，通过 [0] 来获取元组中的第一个元素，即 u 关于 t 的一阶偏导数 u_t
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_fun(self):
        self.optimizer.zero_grad() # 清空优化器中所有参数的梯度信息

        #调用 net_u 方法，使用边界条件的数据点 (self.x_u, self.t_u)
        # 计算神经网络对边界条件的预测值 u_pred
        u_pred = self.net_u(self.x_u, self.t_u) #x_u=(x_1,x_2,...)^T,t_u=(t_1,t_2,...)^T

        # 调用 net_f 方法，使用内部数据点 (self.x_f, self.t_f)
        # 计算物理方程的残差预测值 f_pred
        f_pred = self.net_f(self.x_f, self.t_f)

        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch. mean(f_pred ** 2)

        loss = loss_u + loss_f

        loss.backward() # 进行反向传播，计算损失函数关于神经网络参数的梯度

        self.iter += 1 # 迭代次数加 1，用于记录优化过程中的迭代步数
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' %
                (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss
    
    #训练物理信息神经网络
    def train(self):
        

        #将神经网络模型设置为训练模式
        self.dnn.train()
       
        # backward & optimizer 反向传播和优化 计算损失并进行反向传播，优化器会根据计算得到的梯度更新模型参数
        self.optimizer.step(self.loss_fun)

    # 使用训练好的物理信息神经网络进行预测
    def predict(self, X):

        # 从输入数据 X 中提取第一列数据作为 x 坐标，并将其转换为可求导的 PyTorch 张量，同时移动到指定设备上
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        # 将神经网络模型设置为评估模式
        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        
        #需要先移动到cpu上 再转换为numpy数组，在后续绘图才不会出错
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f



if __name__ == '__main__':
    nu = 0.01/np.pi
    noise = 0.0
    
    #边界点和内部点个数
    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    #加载数据格式为t:100*1,x:256*1,usol:256*100
    data = scipy.io.loadmat('burgers_shock.mat')

    t = data['t'].flatten()[:, None] #将其转换为列向量
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T #100*256并提取实部

    # 生成 x 和 t 的网格点
    X, T = np.meshgrid(x, t)
    #生成两个形状为 (len(t), len(x)) 的二维数组 X 和 T。X 的每一行对应 x 的所有元素，T 的每一列对应 t 的所有元素

    # 将网格点展平并合并为一个二维数组，作为输入数据
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))#flatten转化为一维数组再升维为列向量
    #将 X 展开后的列向量和 T 展开后的列向量在列方向上拼接起来 (x_1,t_1;x_2,t_1;...) (100*256,2)
    
    u_star = Exact.flatten()[:, None] #将精确解其转换为列向量      

    # 边界
    lb = X_star.min(0) #取每一列的最小值 0是维度为行 1*2
    ub = X_star.max(0)

    # 提取 t = 0 ,x=-1,1边界的输入数据和精确解
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T  #100个时间点的第一行 t=0的所有x点

    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))     # X = -1 
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))     # X = 1 
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3]) #垂直堆叠三个边界数组

    X_f_train = lb + (ub-lb)*lhs(2, N_f)# 使用拉丁超立方抽样（LHS）生成内部数据点,二维数据点

    X_f_train = np.vstack((X_f_train, X_u_train)) # 将边界条件数据点添加到内部数据点中

    u_train = np.vstack([uu1, uu2, uu3]) # 合并所有边界条件的精确解
    
    # 随机选择 N_u 个边界条件合并的数据点用于训练
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)# 创建 PhysicsInformedNN 模型实例
    #边界条件坐标，对应边界数据精确点，合并后采样点坐标，网络结构，边界条件上下界，粘性系数
    model.train() #训练模型
    
    u_pred, f_pred = model.predict(X_star) #X_star(100*256,2)

    error_u = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)# 计算预测结果与精确解之间的误差，使用L2 - 范数
    print('Error u: %e' % (error_u))

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic') # 将预测结果插值到网格点上
    Error = np.abs(Exact - U_pred)

    # Visualization ##############################################################
    # 0 u(t, x)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        'kx', label='Data (%d points)' % (u_train.shape[0]),
        markersize=4,
        clip_on=False,
        alpha=1.0
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25]*np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50]*np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75]*np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u(t, x)$', fontsize=20)
    ax.tick_params(labelsize=15)
    plt.savefig('burgers_solution.png')
    plt.show()

    # 1 u(t, x) slices

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.set_title('$t = 0.25$', fontsize=15)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50$', fontsize=15)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75$', fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.savefig('burgers_slices.png')
    plt.show()
