import torch
import numpy as np

loss_fn = torch.nn.CrossEntropyLoss()
tensor = torch.from_numpy(np.array([[0, 1, 0, 0, 0]])).float()
idx = torch.from_numpy(np.array([1]))

print(loss_fn(tensor, idx))
