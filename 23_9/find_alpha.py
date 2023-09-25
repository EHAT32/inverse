import torch
import numpy as np
import matplotlib.pyplot as plt

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

def fourier_solution(right_fft : torch.Tensor, kernel_fft : torch.Tensor, m_func : torch.Tensor, alpha  : torch.Tensor) -> torch.Tensor:
    sol_fft = (right_fft * kernel_fft.conj()) / (kernel_fft.conj() * kernel_fft + alpha * m_func)
    return torch.fft.ifft(sol_fft).real

def loss_func(right_fft : torch.Tensor, kernel_fft : torch.Tensor, m_func : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
    pred = fourier_solution(right_fft, kernel_fft, m_func, alpha)
    return torch.norm(pred - target) / torch.norm(target)

f_t = conv(kernel, x_t, t_ax)
noise = np.random.normal(0, 1e-1, len(f_t))


#Фурье от правой части:
kernel_ar = kernel(t_ax)
kernel_fft = np.fft.fft(kernel_ar)
kernel_fft = torch.from_numpy(kernel_fft)
m_omega = lambda omega: omega ** 2 + 1e-4
freq_ax = np.fft.fftfreq(len(t_ax), t_ax[1] - t_ax[0])
m_ar = m_omega(freq_ax)
m_ar = torch.from_numpy(m_ar)

f_t = conv(kernel, x_t, t_ax)
sigma = 1e-1
N = 1000
epoch_num = 100
alpha_opt = np.zeros(N)
target = torch.from_numpy(x_t(t_ax))
for i in range(N):
    noise = np.random.normal(0, sigma, len(f_t))
    f_d = f_t + noise
    F_d = np.fft.fft(f_d)
    F_d = torch.from_numpy(F_d)
    alpha = torch.tensor(1e-9, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=1e-11)
    for j in range(epoch_num):
        loss = loss_func(F_d, kernel_fft, m_ar, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    alpha_opt[i] = alpha.detach().numpy()
    if i % 100 == 0:
        print(f'{i + 1} : {loss}')
np.save('D:/python/inverse/23_9/sigma_1e_1.npy', alpha_opt)
plt.plot(t_ax, target.numpy())
plt.plot(t_ax, fourier_solution(F_d, kernel_fft, m_ar, alpha_opt[-1]).numpy())
plt.show()