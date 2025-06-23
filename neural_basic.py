import mlx.nn as nn
import mlx.core as mx
from typing import Any, Optional, Tuple

class MLP(nn.Module):
    def __init__(self, dim: int, h_dim:int):
        super().__init__()
        self.linear1 = nn.Linear(dim, h_dim)
        self.linear2 = nn.Linear(h_dim, dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x
    
nn_basic = MLP(10, 20)
print(nn)

print(f"result = {nn_basic(mx.ones((10,)))}")