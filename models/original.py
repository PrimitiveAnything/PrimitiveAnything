import torch
import torch.nn as nn
import modules.volumeEncoder as vE
from modules import netUtils
from modules import primitives


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()
        self.ve = vE.convEncoderSimple3d(5, 16, 1, params.useBn)
        outChannels = self.outChannels = self.ve.output_channels
        layers = []
        for i in range(2):
            layers.append(nn.Conv3d(outChannels, outChannels, kernel_size=1))
            layers.append(nn.BatchNorm3d(outChannels))
            layers.append(nn.LeakyReLU(0.2, True))

        self.fc_layers = nn.Sequential(*layers)
        self.fc_layers.apply(netUtils.weightsInit)

        biasTerms = lambda x: 0

        biasTerms.quat = torch.Tensor([1, 0, 0, 0])
        biasTerms.shape = torch.Tensor(params.nz).fill_(-3) / params.shapeLrDecay
        biasTerms.prob = torch.Tensor(1).fill_(0)
        for p in range(len(params.primTypes)):
            if params.primTypes[p] == "Cu":
                biasTerms.prob[p] = 2.5 / params.probLrDecay

        self.primitivesTable = primitives.Primitives(params, outChannels, biasTerms)
        # self.primitivesTable.apply(netUtils.weightsInit)

    def forward(self, x):

        encoding = self.ve(x)
        features = self.fc_layers(encoding)
        primitives, stocastic_actions = self.primitivesTable(features)
        return primitives, stocastic_actions
