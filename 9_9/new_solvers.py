import numpy as np
import matplotlib.pyplot as plt


#----------------простые итерации---------------------
def iter(y, h, x, x_len, kernel, right_part):
    yk = y.copy()
    for i in range(x_len):
        yk[i] = 0
        for j in range(i):
            yk[i] = yk[i] + 2*kernel(x[i], x[j])*y[j]
        yk[i] = yk[i] - kernel(x[i], x[0])*y[0] - kernel(x[i], x[i])*y[i]
        yk[i] = right_part(x[i]) + yk[i]*h/2
    return yk


def solve_iter(kernel, right_part, x, h, eps = 1e-1):
    n = len(x)
    y = right_part(x)
    yk = iter(y, h, x, n, kernel, right_part)
    i = 0
    err = lambda y,yk: np.linalg.norm(y - yk)/np.linalg.norm(y)
    error = err(y,yk)
    while error > eps:
        y = yk.copy()
        yk = iter(y.copy(), h, x, n, kernel, right_part)
        error = err(y, yk)
        i += 1
        if i > 1000: break
    return yk, i

#-----------------квадратурный способ----------------

def quad_solve(kernel, right_part, h, x_ax, iterations, analytical):
    solution = right_part(x_ax)
    error = np.zeros(iterations + 1)
    error[0] = np.abs(np.linalg.norm(solution - analytical(x_ax)) /np.linalg.norm(analytical(x_ax)))
    for j in range(iterations):
        for i in range(1, len(x_ax)):
            solution[i] = (right_part(x_ax[i]) + h / 2 * kernel(x_ax[i], x_ax[0]) * solution[0] + h * np.dot(kernel(x_ax[i],x_ax[1:i-1]), solution[1:i-1])) / (1 - h/2 * kernel(x_ax[i],x_ax[i]))
        error[j + 1] = np.abs(np.mean(solution - analytical(x_ax)))
    return solution, error



# res, i = solve_iter(kernel, right_part, x, h, eps=1e-10)


# quad_solution, _ = quad_solve(kernel, right_part, h, x, 10, analytical)

# plt.plot(x, analytical(x))
# plt.plot(x, quad_solution)
# plt.show()