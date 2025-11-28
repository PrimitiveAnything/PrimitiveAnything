import torch.nn as nn
import pdb

class convEncoderSimple3d(nn.Module):
  def __init__(self, nLayers, nChannelsInit=8, nInputChannels=1, useBn=True):
    super(convEncoderSimple3d, self).__init__()
    nInputChannels = nInputChannels
    nChannelsInit = nChannelsInit
    useBn = useBn
    nOutputChannels = nChannelsInit
    encoder = []

    for i in range(nLayers):
      encoder.append(nn.Conv3d(nInputChannels, nOutputChannels, kernel_size=3, dilation=1, bias=False))
      encoder.append(nn.BatchNorm3d(nOutputChannels))
      encoder.append(nn.LeakyReLU(0.2, True))
      encoder.append(nn.MaxPool3d(kernel_size=2,dilation=2))
      nInputChannels = nOutputChannels
      nOutputChannels = nOutputChannels*2
    self.encoder = nn.Sequential(*encoder)
    self.outputChannels = nOutputChannels//2

  @property
  def output_channels(self):
    return self.outputChannels

  def forward(self, volume):
    x = self.encoder(volume)
    return x
