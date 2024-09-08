from typing import Any
import mlx.core as mlx
import mlx.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, in_dims:int, out_dims:int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dims)
        )
    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = mlx.maximum(x, 0) if i > 0 else x
            x = l(x)
        return x
    
net = NeuralNet(100, 1)
print(net)

params = net.parameters()
print(params['layers']['layers'][4]['weight'].shape)
