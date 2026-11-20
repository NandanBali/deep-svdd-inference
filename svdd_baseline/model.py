from __future__ import annotations

import torch
from torch import nn


class DeepSVDD(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, 32, bias=False),
                                     nn.Linear(32, 16, bias=False), nn.Linear(16, 8, bias=False)])
        self.register_buffer("center", torch.zeros(8))
        for layer in self.layers[:2]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.layers[-1].weight)

    def trace(self, x):
        h1 = torch.relu(self.layers[0](x))
        h2 = torch.relu(self.layers[1](h1))
        z = self.layers[2](h2)
        terms = (z-self.center).square()
        return h1, h2, z, terms, terms.sum(dim=1)

    def forward(self, x):
        return self.trace(x)[2]

    @torch.no_grad()
    def initialize_center(self, normal_x, epsilon=.1, batch_size=4096):
        if not len(normal_x) or epsilon <= 0:
            raise ValueError("Need normal examples and a positive epsilon")
        total = torch.zeros(8, dtype=torch.float64, device=normal_x.device)
        for batch in normal_x.split(batch_size):
            total += self(batch).double().sum(dim=0)
        original = (total/len(normal_x)).float()
        adjusted = torch.where(original.abs() < epsilon,
                               torch.where(original < 0, -epsilon, epsilon), original)
        self.center.copy_(adjusted)
        return original.cpu().tolist()


def objective(model, x, regularization):
    distance = (model(x)-model.center).square().sum(dim=1).mean()
    penalty = regularization/2 * sum(w.square().sum() for w in model.parameters())
    return distance+penalty, distance, penalty
