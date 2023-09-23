import numpy as np

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

