import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.args = args

    def forward(self, obs):
        return obs
