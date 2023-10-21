import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import argrelextrema

def conv(kernel, func):
    result = np.zeros(len(func))
    K = np.flip(kernel, 0)
    for i in range(len(func)):
        result[i] = np.sum(func * np.roll(K, i + 1))
    return result

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

kernel_ar = kernel(t_ax, beta_0)

right_part = conv(kernel_ar, x_t(t_ax))
kernel_fft = np.fft.fft(kernel_ar)
m_omega = lambda omega: omega ** 2 + 1e-4
freq_ax = np.fft.fftfreq(len(t_ax), t_ax[1] - t_ax[0])
m_ar = m_omega(freq_ax)

#generate kernel as matrix
kernel_matr = np.zeros((len(t_ax), len(t_ax)))
ker_flip = np.flip(kernel_ar, 0)
for i in range(len(t_ax)):
    kernel_matr[i] = np.roll(ker_flip, i + 1)
eig_vals = np.linalg.eigvals(kernel_matr)
eig_max = np.max(np.abs(eig_vals))

alpha_max = 2 / eig_max
print(alpha_max)

sigma = 1e-1
noise = np.random.normal(0, sigma)
right_noisy = right_part + noise

def exp(x_ar, alpha, right_noisy, MAX_ITER = 100):
    iter_solution = right_noisy
    res_norm = np.linalg.norm(iter_solution)
    eps = 1e-5
    true_residual = np.zeros(MAX_ITER)
    true_residual[0] = res_norm
    i = 1
    while i < MAX_ITER and true_residual[i] < true_residual[0]:
        iter_solution += alpha * (right_noisy - np.dot(kernel_matr, iter_solution))
        true_residual[i] = (np.linalg.norm(x_ar - iter_solution) / np.linalg.norm(x_ar))**2
        res_norm = np.sqrt(true_residual[i])
        i += 1
    iter_solution = 0
    print(true_residual[0], ' ', true_residual[81])
    return true_residual, i

alpha = 0.03
nam = str(alpha)
nam =  nam.replace('.', '_')
func, i = exp(x_ar, alpha, right_noisy)
func = func[:i]
plt.plot(np.arange(1, i +1), func[:i], label = f'alpha = {alpha}')
np.save('./14_10/'+nam+'.npy', func)
plt.legend()
plt.show()