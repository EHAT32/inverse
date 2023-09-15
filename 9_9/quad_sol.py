import numpy as np
import matplotlib.pyplot as plt

def trapezoid(array, x_ax = np.linspace(0,1)):
    H = (x_ax[-1] - x_ax[0]) / len(x_ax)    # Задаём длину шага H при числе отрезков разбиения N
    sum = 0.5 * (array[0] + array[-1])
    for i in range(1, len(x_ax) - 1):
        sum += array(i)

    return sum

def quad_solver(right_part, kernel, iterations = 1, h = 1e-3):
    solution = right_part
    for _ in range(iterations):
        for i in range(2, len(solution)):
            solution[i] = (right_part[i] + h / 2 * kernel[i, 0] * solution[0] + h * np.dot(kernel[i,1:i-1], solution[1:i-1])) / (1 - h/2 * kernel[i,i])
    return solution
    

x_ax = np.linspace(0,1)
ker = lambda xn, tn : np.exp(xn - tn)
kernel = np.zeros((len(x_ax), len(x_ax)))

for i in range(len(x_ax)):
    for j in range(i + 1):
        kernel[i, j] = ker(x_ax[i], x_ax[j])
        
right_part = np.exp(x_ax)

solution = quad_solver(right_part, kernel, iterations=10)

    
anal_sol = 2 * np.exp(x_ax)

plt.plot(x_ax, anal_sol)
plt.plot(x_ax, solution)
plt.show()
