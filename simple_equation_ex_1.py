import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os


#\frac{\partial^2u}{\partial x^2}-\frac{\partial^4u}{\partial y^4}=(2-x^2)e^{-y}
#uxx(x,0)=x^2,uyy(x,1)=\frac{x^2}{e},u(x,0)=x^2,u(x,1)=\frac{x^2}{e},u(0,y)=0,u(1,y)=e^{-y}

# 检查GPU是否可用并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

epochs = 10000
h = 100
N = 1000
N1 = 100
N2 = 1000


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(888888)


def interior(n=N):
    x = torch.rand(n, 1).to(device)  # 将生成的数据移到GPU上（如果可用）
    y = torch.rand(n, 1).to(device)
    cond = (2 - x ** 2) * torch.exp(-y).to(device)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down_yy(n=N1):
    x = torch.rand(n, 1).to(device)
    y = torch.ones_like(x).to(device)
    cond = x ** 2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up_yy(n=N1):
    x = torch.rand(n, 1).to(device)
    y = torch.ones_like(x).to(device)
    cond = x ** 2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down(n=N1):
    x = torch.rand(n, 1).to(device)
    y = torch.zeros_like(x).to(device)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up(n=N1):
    x = torch.rand(n, 1).to(device)
    y = torch.ones_like(x).to(device)
    cond = x ** 2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond


def left(n=N1):
    y = torch.rand(n, 1).to(device)
    x = torch.zeros_like(y).to(device)
    cond = torch.zeros_like(x).to(device)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def right(n=N1):
    y = torch.rand(n, 1).to(device)
    x = torch.ones_like(y).to(device)
    cond = torch.exp(-y).to(device)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def data_interior(n=N2):
    x = torch.rand(n, 1).to(device)
    y = torch.rand(n, 1).to(device)
    cond = (x ** 2) * torch.exp(-y).to(device)
    return x.requires_grad_(True), y.requires_grad_(True), cond


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


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True,)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


# 以下7个损失是PDE损失，修改了每个损失函数计算的方式，使其符合MSELoss的参数要求
def l_interior(u):
    # 损失函数L1
    x, y, cond = interior()
    uxy = u(torch.cat([x, y], dim=1))
    # 这里假设cond不是直接参与loss计算，而是作为一种条件判断等用途（可根据实际调整）
    # 计算预测值相关的梯度，将其作为MSELoss的第一个参数（预测值）
    pred_value = gradients(uxy, x, 2) + gradients(uxy, y, 4)
    # 目标值这里暂时简单设为全0（可根据实际物理意义等修改为正确的目标值）
    target_value = torch.zeros_like(pred_value)
    return loss(pred_value, target_value)


def l_down_yy(u):
    # 损失函数L2
    x, y, cond = down_yy()
    uxy = u(torch.cat([x, y], dim=1))
    pred_value = gradients(uxy, x, 2)
    target_value = torch.zeros_like(pred_value)
    return loss(pred_value, target_value)


def l_up_yy(u):
    # 损失函数L3
    x, y, cond = up_yy()
    uxy = u(torch.cat([x, y], dim=1))
    pred_value = gradients(uxy, x, 2)
    target_value = torch.zeros_like(pred_value)
    return loss(pred_value, target_value)


def l_down(u):
    # 损失函数L4
    x, y, cond = down()
    uxy = u(torch.cat([x, y], dim=1))
    pred_value = uxy
    target_value = torch.zeros_like(pred_value)
    return loss(pred_value, target_value)


def l_up(u):
    # 损失函数L5
    x, y, cond = up()
    uxy = u(torch.cat([x, y], dim=1))
    pred_value = uxy
    target_value = torch.zeros_like(pred_value)
    return loss(pred_value, target_value)


def l_left(u):
    # 损失函数L6
    x, y, cond = left()
    uxy = u(torch.cat([x, y], dim=1))
    pred_value = uxy
    target_value = torch.zeros_like(pred_value)
    return loss(pred_value, target_value)


def l_right(u):
    # 损失函数L7
    x, y, cond = right()
    uxy = u(torch.cat([x, y], dim=1))
    pred_value = uxy
    target_value = torch.zeros_like(pred_value)
    return loss(pred_value, target_value)


def l_data(u):
    # 损失函数L8
    x, y, cond = data_interior()
    uxy = u(torch.cat([x, y], dim=1))
    pred_value = uxy
    target_value = cond
    return loss(pred_value, target_value)


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
# Training
u = MLP().to(device)  # 将模型移到GPU上（如果可用）
opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)  # 学习率调整策略，每5000轮次学习率减半
start_time = time.time()  # 记录训练开始时间
for i in range(epochs):
    opt.zero_grad()
    l = (l_interior(u) + l_up_yy(u) + l_down_yy(u) + l_up(u) + l_down(u) + l_left(u) + l_right(u) + l_data(u)).to(device)
    l.backward()
    opt.step()
    scheduler.step()  # 更新学习率
    if i % 1000 == 0 and i > 0:  # 这里添加了i > 0的判断，避免i为0时调用出现除以0错误
        print_time_cost(start_time, i, epochs)
    if i % 100 == 0:
        print(i)
end_time = time.time()  # 记录训练结束时间
print(f"Training time: {end_time - start_time} seconds")

# 保存训练好的模型
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(u.state_dict(),'models/model.pth')

# 加载模型进行推理（可以注释掉上面的训练部分，直接加载已保存的模型进行推理）
# u = MLP().to(device)
# u.load_state_dict(torch.load('models/model.pth'))

# Inference
xc = torch.linspace(0, 1, h).to(device)
xm, ym = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1).to(device)
yy = ym.reshape(-1, 1).to(device)
u_pred = u(torch.cat([xx, yy], dim=1))
u_real = xx * xx * torch.exp(-yy).to(device)
u_error = torch.abs(u_pred - u_real)
u_pred_fig = u_pred.reshape(h, h).cpu().detach().numpy()
u_real_fig = u_real.reshape(h, h).cpu().detach().numpy()
u_error_fig = u_error.reshape(h, h).cpu().detach().numpy()
print("Max abs error is: ", float(torch.max(torch.abs(u_pred - xx * xx * torch.exp(-yy)))))
print(xx)
print(yy)

# 作PINN数值解图
fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot_surface(xm.detach().cpu().numpy(), ym.detach().cpu().numpy(), u_pred_fig)
ax.text2D(0, 0.9, "PINN", transform=ax.transAxes)
plt.show()
fig.savefig("PINN_solve.png")

# 作真解图
fig = plt.figure(2)
ax = Axes3D(fig)
ax.plot_surface(xm.detach().cpu().numpy(), ym.detach().cpu().numpy(), u_real_fig)
ax.text2D(0, 0.9, "real solve", transform=ax.transAxes)
plt.show()
fig.savefig("real_solve.png")

# 误差图
fig = plt.figure(3)
ax = Axes3D(fig)
ax.plot_surface(xm.detach().cpu().numpy(), ym.detach().cpu().numpy(), u_error_fig)
ax.text2D(0, 0.9, "abs error", transform=ax.transAxes)
plt.show()
fig.savefig("abs_error.png")