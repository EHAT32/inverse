import numpy as np
import torch
import matplotlib.pyplot as plt

def fourier_solve(kernel_fft, right_fft, m_omega, alpha):
    sol_fft = (right_fft * kernel_fft.conj()) / (kernel_fft.conj() * kernel_fft + np.abs(alpha) * m_omega)
    return np.fft.ifft(sol_fft).real

def conv(kernel, func):
    result = np.zeros(len(func))
    F = np.flip(func, 0)
    for i in range(len(func)):
        result[i] = np.sum(kernel * np.roll(F, i + 1))
    return result

def conv_2(kernel, func):
    result = np.zeros(len(func))
    K = np.flip(kernel, 0)
    for i in range(len(func)):
        result[i] = np.sum(func * np.roll(K, i + 1))
    return result

def torch_conv(kernel : torch.Tensor, func : torch.Tensor) -> torch.Tensor:
    result = torch.zeros(len(kernel))
    F = torch.flip(func, (0,))
    for i in range(len(F)):
        result[i] = torch.sum(kernel * torch.roll(F, i + 1))
    return result

def loss_func(kernel_fft : torch.Tensor, right_fft : torch.Tensor, m_omega : torch.Tensor, alpha : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
    sol = fourier_solve(kernel_fft, right_fft, m_omega, alpha)
    return torch.linalg.norm(sol - target) / torch.linalg.norm(target)

def loss_2(kernel : torch.Tensor, kernel_fft : torch.Tensor, right_fft : torch.Tensor, m_omega : torch.Tensor, alpha : torch.Tensor, sigma : torch.Tensor) -> torch.Tensor:
    sol = fourier_solve(kernel_fft, right_fft, m_omega, alpha)
    return torch.linalg.norm(torch_conv(kernel, sol) - right_fft)**2 - sigma**2
    
f_0 = 300
mu = 8e5
a = 0
b = 1e-2
N = 100
t_ax = np.linspace(a,b,N)

x_t = lambda t : np.cos(2 * np.pi * f_0 * t + mu * t**2)
x_ar = x_t(t_ax)

beta_0 = 3e3
A = 0.5
dt = 2e-3

kernel = lambda t, beta : np.exp(- beta * t) + A * np.exp(- beta * t - dt)
kernel_ar = kernel(t_ax, beta_0 * 1)
f_t = conv(kernel_ar, x_ar)


# #Фурье от правой части:
kernel_fft = np.fft.fft(kernel_ar)
m_omega = lambda omega: omega ** 2 + 1e-4
freq_ax = np.fft.fftfreq(len(t_ax), t_ax[1] - t_ax[0])
m_ar = m_omega(freq_ax)


kernel_ar = torch.from_numpy(kernel_ar)
kernel_fft = torch.from_numpy(kernel_fft)
m_ar = torch.from_numpy(m_ar)
target = torch.from_numpy(x_ar)

epoch_num = 100
alpha_opt = np.zeros(epoch_num)

sigma = 1e-1

# alpha_range = np.linspace(1e-15, 1e-13, 100)
# for i in range(epoch_num):
#     noise = np.random.normal(0, sigma)
#     sol = np.zeros(epoch_num)
#     right_noisy = f_t + noise
#     right_fft = np.fft.fft(right_noisy)
#     for j in range(100):
#         solut = fourier_solve(kernel_fft, right_fft, m_ar, alpha_range[j])
#         sol[j] = np.linalg.norm(conv(kernel_ar, solut) - right_noisy)-sigma
#     alpha_opt[i] = alpha_range[np.argmin(sol)]
#     if i % 50 == 0:
#         print(i, ' ', alpha_opt[i])

# alpha = torch.tensor(1e-8, dtype=torch.float32, requires_grad=True)
# for j in range(epoch_num):
#     optimizer = torch.optim.Adam([alpha], lr=1e-11, weight_decay=0.8)
#     noise = np.random.normal(0, sigma, len(f_t))
#     f_d = f_t + noise
#     F_d = torch.from_numpy(np.fft.fft(f_d))
#     sigma = torch.tensor(sigma * 0.95, dtype=torch.float32)
#     for i in range(50):
#         loss = loss_2(kernel_ar, kernel_fft, F_d, m_ar, alpha, sigma)
#         # loss = loss_func(kernel_fft, F_d, m_ar, alpha, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # if alpha.item() < 1e-15:
#         #     alpha = - alpha
#     if j % 10 == 0:
#         print(f'{j + 1} : {alpha.item()}')
#     alpha_opt[j] = alpha.detach().numpy()
# np.save('C:/python/inverse/30_9/sec_met_1_1_low_n_back.npy', alpha_opt)
# print(alpha_opt)

# plt.hist(alpha_opt)
# plt.show()
