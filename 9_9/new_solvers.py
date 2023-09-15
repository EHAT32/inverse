import numpy as np
import matplotlib.pyplot as plt

def iter(y, h, x, n, k, f):
    yk = y.copy()
    for i in range(n):
        yk[i] = 0
        for j in range(i):
            yk[i] = yk[i] + 2*k(x[i], x[j])*y[j]
        yk[i] = yk[i] - k(x[i], x[0])*y[0] - k(x[i], x[i])*y[i]
        yk[i] = f(x[i]) + yk[i]*h/2
    return yk


def solve_iter(k, f, x, h, eps =1e-1):
    n = len(x)
    y = f(x)
    yk = iter(y, h, x, n, k, f)
    i = 0
    err = lambda y,yk: np.linalg.norm(y - yk)/np.linalg.norm(y)
    error = err(y,yk)
    #print(max(y-yk))
    while error > eps:
        #print(error(y, yk))
        y = yk.copy()
        yk = iter(y.copy(), h, x, n, k, f)
        #print(error(y, yk))
        error = err(y, yk)
        i += 1
        if i > 1000: break
    return yk, i

a = 0
b = 1
h = 1e-2
l = 1
beta = -1

N = int((b-a)/h)

k_func = lambda x, t: np.exp(l*(x-t))
f_func = lambda t: np.exp(beta*t)

x = np.linspace(a, b, N)
x_ex = [2*np.exp(x[i])-1 for i in range(N)]
res, i = solve_iter(k_func, f_func, x, h, eps=1e-10)

analytical = 2 * np.exp(x)

plt.plot(x, analytical)
plt.plot(x, res)
plt.show()