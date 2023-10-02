import numpy as np
import torch

a = np.array([1, 2, 3, 4, 5, 6])
a = torch.from_numpy(a)
print(np.sum(a, 0))