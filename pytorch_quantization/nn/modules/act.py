import torch

class HardswishReplace(torch.nn.Module):
    def __init__(self):
        super(HardswishReplace, self).__init__()
        self.gate = torch.nn.Hardsigmoid()
    def forward(self, x):
        return x * self.gate(x)

