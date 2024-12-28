import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# 检查GPU是否可用并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 训练轮次、网格点数等参数设置
epochs =10000
h = 100
N = 1000  # 内部采样点数
N1 = 100  # 边界采样点数
N2 = 1000  # 数据点采样点数

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(88888)

# 定义二维坐标生成函数（内部点）
def interior(n=N):
    x = torch.rand(n, 1).to(device)
    y = torch.rand(n, 1).to(device)
    return x.requires_grad_(True), y.requires_grad_(True)

# 定义边界条件相关函数（例如这里简单假设边界上函数值为0，可根据实际修改）
def boundary(n=N1):
    x = torch.rand(n, 1).to(device)
    y = torch.zeros_like(x).to(device)  # 这里以x轴边界为例，可扩展到其他边界
    return x.requires_grad_(True), y.requires_grad_(True)

# 定义源项函数f(x,y)，这里示例为一个简单的函数，可根据实际泊松方程的情况修改
def source_term(x, y):
    return 2 * torch.sin(x) * torch.cos(y)  # 示例源项函数，可替换

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


loss = torch.nn.MSELoss()

# 计算梯度的函数
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True,)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# 内部点损失函数（基于泊松方程的形式）
def l_interior(u):
    x, y = interior()
    uxy = u(torch.cat([x, y], dim=1))
    laplacian_uxy = gradients(gradients(uxy, x, 2), y, 2)  # 计算拉普拉斯算子
    f_value = source_term(x, y)
    return loss(laplacian_uxy, f_value)

# 边界条件损失函数（这里假设边界上函数值为0）
def l_boundary(u):
    x, y = boundary()
    uxy = u(torch.cat([x, y], dim=1))
    target_value = torch.zeros_like(uxy)
    return loss(uxy, target_value)


# 定义计时函数
def print_time_cost(start_time, current_epoch, total_epochs):
    elapsed_time = time.time() - start_time
    current_hours = int(elapsed_time // 3600)
    current_minutes = int((elapsed_time % 3600) // 60)
    current_seconds = int(elapsed_time % 60)
    total_hours = int((elapsed_time / current_epoch * total_epochs) // 3600)
    total_minutes = int(((elapsed_time / current_epoch * total_epochs) % 3600) // 60)
    total_seconds = int((elapsed_time / current_epoch * total_epochs) % 60)
    print(f"Epoch: {current_epoch}, Current time cost: {current_hours}h {current_minutes}m {current_seconds}s, "
          f"Total estimated time cost: {total_hours}h {total_minutes}m {total_seconds}s")


# Training
u = MLP().to(device)
opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)
start_time = time.time()
for i in range(epochs):
    opt.zero_grad()
    l = l_interior(u) + l_boundary(u)  # 综合内部点和边界条件损失
    l.backward()
    opt.step()
    scheduler.step()
    if i % 1000 == 0 and i > 0:
        print_time_cost(start_time, i, epochs)
    if i % 100 == 0:
        print(i)
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

# 保存训练好的模型
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(u.state_dict(),'models/model.pth')

# 加载模型进行推理（可以注释掉上面的训练部分，直接加载已保存的模型进行推理）
# u = MLP().to(device)
# u.load_state_dict(torch.load('models/model.pth'))

# 生成网格用于可视化
xc = torch.linspace(0, 1, h).to(device)
xm, ym = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1).to(device)
yy = ym.reshape(-1, 1).to(device)
u_pred = u(torch.cat([xx, yy], dim=1))
u_pred_fig = u_pred.reshape(h, h).cpu().detach().numpy()

# 作PINN数值解图
fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot_surface(xm.detach().cpu().numpy(), ym.detach().cpu().numpy(), u_pred_fig)
ax.text2D(0, 0.9, "PINN", transform=ax.transAxes)
plt.show()
fig.savefig("PINN_solve.png")