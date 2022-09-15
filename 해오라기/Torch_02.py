import torch
import torch.nn as nn # 
import torch.nn.functional as F # 
import torch.optim as optim
import numpy as np
 
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

x = torch.FlaotTensor(x).unsqueeze(-1) # unsqueeze(1) 이어도 가능
y = torch.FlaotTensor(x).unsqueeze(-1)
print(x, y)
print(x.shape, y.shape)

model = nn.Linear(3, )