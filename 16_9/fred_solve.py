import numpy as np
import matplotlib.pyplot as plt
# def iter(y, h, x, x_len, kernel, right_part):
#     yk = y.copy()
#     for i in range(x_len):
#         yk[i] = 0
#         for j in range(x_len):
#             yk[i] = yk[i] + 2*kernel(x[i], x[j])*y[j]
#         yk[i] = yk[i] - kernel(x[i], x[0])*y[0] - kernel(x[i], x[i])*y[i]
#         yk[i] = right_part(x[i]) + yk[i]*h/2
#     return yk


# def solve_iter(kernel, right_part, x, h, eps =1e-1):
#     n = len(x)
#     y = right_part(x)
#     yk = iter(y, h, x, n, kernel, right_part)
#     i = 0
#     error = []
#     err = lambda y,yk: np.linalg.norm(y - yk)/np.linalg.norm(y)
#     error.append(err(y,yk))
#     while error[-1] > eps:
#         y = yk.copy()
#         yk = iter(y.copy(), h, x, n, kernel, right_part)
#         error.append(err(y, yk))
#         i += 1
#         if i > 1000: break
#     return yk, error

# def quad_solve(right_part, kernel, x, eps = 1e-1):
#     solution = right_part
#     h = (x[-1] - x[0]) / len(x)
#     error = []
#     y = right_part
#     error.append(1)
#     iter = 0
#     while error[-1] > eps:
#         y = solution.copy()
#         for i in range(1, len(solution)):
#             solution[i] = (right_part[i] + h / 2 * kernel[i, 0] * solution[0] + h * np.dot(kernel[i,1:i-1], solution[1:i-1])) / (1 - h/2 * kernel[i,i])
#         error.append(np.linalg.norm(solution - analytical) / np.linalg.norm(analytical))
#         iter += 1
#         if iter > 1000:
#             break
#     return solution, error
    

# x_ax = np.linspace(0,1)
# h = (x_ax[-1] - x_ax[0]) / len(x_ax)
# #зададим ядро
# ker = lambda xn, tn : np.exp(xn - tn)
# kernel = np.zeros((len(x_ax), len(x_ax)))
# for i in range(len(x_ax)):
#     for j in range(len(x_ax)):
#         kernel[i, j] = ker(x_ax[i], x_ax[j])
# right_part = lambda x : np.exp(x)
# analytical = np.exp(x_ax) * 10

# quad_solution, error = quad_solve(right_part(x_ax), kernel, x_ax)
# # iter_solution, error = solve_iter(ker, right_part, x_ax, h)
# plt.figure(1)
# plt.plot(x_ax, quad_solution)
# plt.plot(x_ax, analytical)

# plt.figure(2)
# plt.scatter(np.arange(len(error)), error)
# plt.show()

def quad_fr(k_func, f_func, a, b, h):
    x = np.linspace(a, b, int((b-a)/h))
    n = len(x)
    wt = 0.5
    wj = 1
    a = np.zeros((n,n), dtype=float)
    for i in range(n):
        a[i, 0] = -h*wt*k_func(x[i], x[0])
        for j in range(1, n-1):
            a[i,j] = -h*wj*k_func(x[i], x[j])
        a[i,n-1] = -h*wt*k_func(x[i], x[n-1])
        a[i,i] += 1.
        b = np.zeros_like(x)
        for j in range(n):
            b[j] = f_func(x[j])
    y = np.linalg.solve(a, b)
    return y

