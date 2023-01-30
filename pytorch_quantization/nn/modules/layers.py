import torch
import torch.nn as nn

class HardswishReplace(nn.Module):
    def __init__(self):
        super(HardswishReplace, self).__init__()
        self.gate = nn.Hardsigmoid()
    def forward(self, x):
        return x * self.gate(x)

class FTMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.bias1 = nn.Parameter(torch.Tensor(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.bias2 = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bias1 + x
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bias2 + x
        x = self.drop2(x)
        return x
