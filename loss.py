import torch.nn as nn
import torch

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce = nn.CrossEntropyLoss()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, pred, target):
        loss = self.bce(pred, target)
        return loss