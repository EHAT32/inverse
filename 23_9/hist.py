import matplotlib.pyplot as plt
import numpy as np

sigma_1e_1 = np.load('D:/python/inverse/23_9/sigma_1e_1.npy')
sigma_6e_2 = np.load('D:/python/inverse/23_9/sigma_6e_2.npy')

left = min(np.min(sigma_1e_1), np.min(sigma_6e_2))
right = max(np.max(sigma_1e_1), np.max(sigma_6e_2))

plt.figure(1)
plt.hist(sigma_1e_1, bins = 100)
plt.xlabel('alpha')
plt.ylabel('Frequency')
plt.xlim((np.min(sigma_1e_1)), np.max(sigma_1e_1) * 1.001)
plt.figure(2)
plt.hist(sigma_6e_2, bins = 100)
plt.xlabel('alpha')
plt.ylabel('Frequency')
plt.xlim((np.min(sigma_6e_2)), np.max(sigma_6e_2))

# plt.figure(3)
# plt.hist(sigma_1e_1, bins = 100, label='$\sigma_1 = 1e-1$')
# plt.hist(sigma_6e_2, bins=100, label='$\sigma_2 = 6e-2$')
# plt.xlim((left, right))
# plt.xlabel('alpha')
# plt.ylabel('Frequency')

plt.show()