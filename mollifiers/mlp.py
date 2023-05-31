import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class TwoHiddenLayerMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
class ClippedModule(nn.Module):
    def __init__(self, module: nn.Module, clip: float) -> None:
        super().__init__()
        
        self.module = module
        self.clip = clip
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(self.module(x), min = -self.clip, max = self.clip)