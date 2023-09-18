import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def iter_solve(right_part, kernel, iterations = 1, dx = 1e-3, x = np.linspace(0,1, 50), analytical = None):
    solution = right_part
    error = np.zeros(iterations + 1)
    error[0] = np.abs(np.mean(solution - analytical))
    for j in range(iterations):
        for i in range(1, len(x)):
            solution[i] = right_part[i] + np.trapz(kernel[i, :i] * solution[:i], x[:i])
        error[j + 1] = np.abs(np.mean(solution - analytical - 1))
    return solution, error

def quad_solve(right_part, kernel, iterations = 1, h = 1e-3, analytical = None):
    solution = right_part
    error = np.zeros(iterations + 1)
    error[0] = np.linalg.norm(solution - analytical) / np.linalg.norm(analytical)
    for j in range(iterations):
        for i in range(1, len(solution)):
            solution[i] = (right_part[i] + h / 2 * kernel[i, 0] * solution[0] + h * np.dot(kernel[i,1:i-1], solution[1:i-1])) / (1 - h/2 * kernel[i,i])
        error[j + 1] = np.linalg.norm(solution - analytical) / np.linalg.norm(analytical)
    
    return solution, error

def iter(y, h, x, x_len, kernel, right_part):
    yk = y.copy()
    for i in range(x_len):
        yk[i] = 0
        for j in range(i):
            yk[i] = yk[i] + 2*kernel(x[i], x[j])*y[j]
        yk[i] = yk[i] - kernel(x[i], x[0])*y[0] - kernel(x[i], x[i])*y[i]
        yk[i] = right_part(x[i]) + yk[i]*h/2
    return yk


def solve_iter(kernel, right_part, x, h, eps =1e-1):
    n = len(x)
    y = right_part(x)
    yk = iter(y, h, x, n, kernel, right_part)
    i = 0
    error = []
    err = lambda y,yk: np.linalg.norm(y - yk)/np.linalg.norm(y)
    error.append(err(y,yk))
    while error[-1] > eps:
        y = yk.copy()
        yk = iter(y.copy(), h, x, n, kernel, right_part)
        error.append(err(y, yk))
        i += 1
        if i > 1000: break
    return yk, error

# x_ax = np.linspace(0,1)

# #зададим ядро
# ker = lambda xn, tn : np.exp(xn - tn)
# kernel = np.zeros((len(x_ax), len(x_ax)))

# for i in range(len(x_ax)):
#     for j in range(i + 1):
#         kernel[i, j] = ker(x_ax[i], x_ax[j])
        
# #зададим правую часть
# right_part_l = lambda x: np.exp(x)
# right_part = np.exp(x_ax)

# #аналитическое решение
# analytical = 2*np.exp(x_ax) - 1

# #решим квадратурно
# iterations = 10
# quad_solution, _ = quad_solve(right_part, kernel, iterations=iterations, analytical=analytical)

# plt.figure(1)
# plt.subplot(1,2, 1)
# plt.title(f'Решение квадратурным способом ({iterations} итераций)')
# plt.plot(x_ax, quad_solution, label='численно')
# plt.plot(x_ax, analytical, label='точное решение')
# plt.legend()

# #ошибка от числа итераций
# iterations = 100
# _, quad_error = quad_solve(right_part, kernel, iterations=iterations, analytical=analytical)

# plt.subplot(1,2, 2)
# plt.plot(range(iterations + 1), quad_error)
# plt.xlabel('итерация')
# plt.ylabel('абсолютная ошибка')

# #решим простыми итерациями
# # iter_solution = iter_solve(right_part, kernel, iterations=1, analytical=analytical)
# iter_solution, _ = iter_solve(ker, right_part_l, x_ax, h=1e-3)

# plt.figure(2)
# plt.subplot(1,2, 1)
# plt.title('Решение простыми итерациями (одна итерация)')
# plt.plot(x_ax, iter_solution, label='численно')
# plt.plot(x_ax, analytical, label='точное решение')
# plt.legend()

# iterations = 10
# _, iter_error = iter_solve(right_part, kernel, iterations=iterations, analytical=analytical)
# plt.subplot(1,2,2)
# plt.plot(range(iterations + 1), iter_error)
# plt.xlabel('итерация')
# plt.ylabel('абсолютная ошибка')

# # Рассмотрим квадратурный метод, так как он реализован лучще всего.
# # Зафиксируем число итераций, когда ошибка минимальна, будем добавлять шум с разным отклонением и смотреть на ошибку
# # Будем менять отклонение в диапазоне от 0 до 0.1
# n = 20
# noise_error = np.empty(n)
# for i in range(n):
#     noise = np.random.normal(0,i*1e-1, len(right_part))
#     quad_solution, err = quad_solve(right_part + noise, kernel, iterations=18, analytical=analytical)
#     noise_error[i] = err[-1]

# plt.figure(3)
# plt.title('Ошибка от с.о. шума')
# plt.plot([k*1e-1 for k in range(n)], noise_error)
# plt.xlabel('с.о. шума')
# plt.ylabel('абсолютная ошибка')

# plt.show()