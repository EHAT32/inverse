import numpy as np
import torch

def fourier_solve(kernel_fft : torch.Tensor, right_fft : torch.Tensor, m_omega : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
    sol_fft = (right_fft * kernel_fft.conj()) / (kernel_fft.conj() * kernel_fft + alpha * m_omega)
    return torch.fft.ifft(sol_fft).real

def torch_conv(kernel : torch.Tensor, func, t_ax) -> torch.Tensor:
    result = torch.zeros(len(t_ax))
    K = kernel(t_ax)
    F = func(t_ax)[::-1]
    for i in range(len(t_ax)):
        result[i] = np.sum(K * np.roll(F, i + 1))
    return result

def loss_func(kernel_fft : torch.Tensor, right_fft : torch.Tensor, m_omega : torch.Tensor, alpha : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
    sol = fourier_solve(kernel_fft, right_fft, m_omega, alpha)
    return torch.linalg.norm(sol - target) / torch.linalg.norm(target)

def loss_2(kernel : torch.Tensor, kernel_fft : torch.Tensor, right_fft : torch.Tensor, m_omega : torch.Tensor, alpha : torch.Tensor, sigma : torch.Tensor) -> torch.Tensor:
    t_ax = torch.linspace(0, 1e-2, len(kernel))
    sol = fourier_solve(kernel_fft, right_fft, m_omega, alpha)
    return torch.linalg.norm(torch_conv(kernel, sol, t_ax) - right_fft)**2 - sigma**2
    

f_0 = 300
mu = 8e5
a = 0
b = 1e-2
N = 100
t_ax = np.linspace(a,b,N)

x_t = lambda t : np.cos(2 * np.pi * f_0 * t + mu * t**2)

beta = 3e3
A = 0.5
dt = 2e-3

kernel = lambda t : np.exp(- beta * t) + A * np.exp(- beta * t - dt)

def conv(kernel, func, t_ax):
    result = np.zeros(len(t_ax))
    K = kernel(t_ax)
    F = func(t_ax)[::-1]
    for i in range(len(t_ax)):
        result[i] = np.sum(K * np.roll(F, i + 1))
    return result

f_t = conv(kernel, x_t, t_ax)

kernel_ar = kernel(t_ax)
kernel_ar = torch.from_numpy(kernel_ar)
kernel_fft = torch.from_numpy(np.fft.fft(kernel_ar))
m_omega = lambda omega: omega ** 2 + 1e-4
freq_ax = np.fft.fftfreq(len(t_ax), t_ax[1] - t_ax[0])

m_ar = torch.from_numpy(m_omega(freq_ax))
x_ar = torch.from_numpy(x_t(t_ax))
exp_num = 1000
alpha_opt = np.zeros(exp_num)
sigma = 6e-2
for j in range(exp_num):
    alpha = torch.tensor(1e-9, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=1e-11)
    noise = np.random.normal(0, sigma, len(f_t))
    f_d = f_t + noise
    F_d = torch.from_numpy(np.fft.fft(f_d))
    for i in range(100):
        loss = loss_func(kernel_fft, F_d, m_ar, alpha, x_ar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    alpha_opt[j] = alpha.detach().numpy()
    if j % 100 == 0:
        print(f'{j + 1} : {loss}')
np.save('C:/python/inverse/23_9/sigma_6_e_2_new_step.npy', alpha_opt)