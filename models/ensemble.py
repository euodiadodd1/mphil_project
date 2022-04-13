import torch
import torch.nn as  nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class Ensemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(15, 5)
        
    def forward(self, x1, x2, x3):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x3 = self.modelC(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(F.relu(x))
        return x

class AvgEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(AvgEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        #self.classifier = nn.Linear(15, 5)
        
    def forward(self, x1, x2, x3):
        x1 = self.modelA(x1).detach().cpu().numpy()
        x2 = self.modelB(x2).detach().cpu().numpy()
        x3 = self.modelC(x3).detach().cpu().numpy()
        #x = torch.cat((x1, x2, x3), dim=1)

        avg = [np.mean([x1[:,i], x2[:,i], x3[:,i]], axis=0) for i in range(x1.shape[1])]
        x = torch.tensor(np.column_stack(avg), requires_grad=True).float()

        return x
