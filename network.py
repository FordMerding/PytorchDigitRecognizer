import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax(1)
        )
    def forward(self, a):
        x = self.flatten(a)
        out = self.layers(x)
        return out
    