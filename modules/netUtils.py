import torch
import pdb

def weightsInit(m):
  name = str(type(m))
  if 'Conv3d' in name:
    m.weight.data = m.weight.data.normal_(mean=0,std=0.02)
    m.bias.data = m.bias.data.zero_()
  elif 'BatchNorm3d' in name:
    m.weight.data = m.weight.data.normal_(mean=1.0, std=0.02)
    m.bias.data = m.bias.data.zero_()

def scaleWeightsFunc(scale, key):
  def internal_scale(m):
    # name = str(type(m))
    try:
      if m.note in key:
        print(key)
        m.weight.data = m.weight.data.mul_(scale)
        m.bias.data = m.bias.data.mul_(scale)
    except AttributeError:
      assert True
  return internal_scale


def scaleBiasWeights(scale, key):
  def internal_scale(m):
    # name = str(type(m))
    try:
      if m.note in key:
        m.bias.data = m.bias.data.fill_(2.5/scale)
    except AttributeError:
      assert True
  return internal_scale