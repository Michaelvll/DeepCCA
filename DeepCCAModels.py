import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):
    # TODO: Maybe need to add l2 loss in loss
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                )
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, reg_par=0, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()
        self.reg_par = reg_par

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
    
    @DeprecationWarning
    def _loss(self, o1, o2):
        model_loss = self.loss(o1, o2)
        reg_loss = torch.Tensor([0.0])
        reg_par = self.reg_par
        if reg_par:
            for W in self.model1.parameters():
                reg_loss = reg_loss + reg_par * W.norm(2)
            for W in self.model2.parameters():
                reg_loss = reg_loss + reg_par * W.norm(2)
        loss = reg_loss + model_loss
        return loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2
