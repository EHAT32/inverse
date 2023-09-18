import numpy as np

def iter(y, h, x, x_len, kernel, right_part):
    yk = y.copy()
    for i in range(x_len):
        yk[i] = 0
        for j in range(x_len):
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