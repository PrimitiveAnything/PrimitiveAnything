import sys
import torch
sys.path.insert(0, '/home/nileshk/Research2/volumetricPrimitivesPytorch/')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from modules.quatUtils import  quat_conjugate
from modules.transformer import rotate_module, translate_module
from torch.autograd import Variable

def plot3(points, mark="o", col="r", block=True):
  # points is a torch.FloatTensor.
  points_numpy = points.numpy()
  x = points_numpy[:, 0].squeeze()
  y = points_numpy[:, 1].squeeze()
  z = points_numpy[:, 2].squeeze()
  fig = plt.figure()

  ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=112, azim=-97)
  ax.scatter(x, y, zs=z, zdir='z', marker=mark, color=col)
  plt.show(block = block)

cube = 2*torch.Tensor([[0,0,0], [0 ,1, 0],[0, 1,1],[0, 0, 1],
                 [1,0,1],[1,1,1],[1,1,0],[1,0,0]])  - 1
# R'*p_in + t
import pdb
def plot_cuboid(part, ax, color):
  shape = part[0:3].view(1,1,3)
  trans = part[3:6].view(1,1,3)
  quat = part[6:10].view(1,1,4)

  shape_rep = shape.repeat(1, 8,1)

  scale_points = cube*shape_rep
  scale_points = scale_points.unsqueeze(0)
  quat_c = quat_conjugate(quat)
  rotated_points = rotate_module(scale_points, quat_c)
  trans_points = translate_module(rotated_points, trans)
  draw_points(ax, color, trans_points)


def plot_parts(parts,block=True):
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_xlim(-0.7, 0.7)
  ax.set_ylim(-0.7, 0.7)
  ax.set_zlim(-0.7, 0.7)

  parts = parts.cpu().data
  cmap = plt.get_cmap('jet_r')
  parts = torch.chunk(parts, chunks=parts.size(0), dim=0)
  for i in range(len(parts)):
    plot_cuboid(parts[i].view(-1), ax, cmap(float(i)/len(parts)))

  plt.show(block=block)

connections = []
connections.append([0,1])
connections.append([1,2])
connections.append([2,3])
connections.append([0,3])

connections.append([3,4])
connections.append([2,5])
connections.append([1,6])
connections.append([0,7])


connections.append([4,5])
connections.append([5,6])
connections.append([6,7])
connections.append([4,7])

def draw_points(ax, color, ps):

  numpy_points = ps.numpy().squeeze()
  for i in range(12):
    s = connections[i][0]
    t = connections[i][1]
    ax.plot([numpy_points[s,0], numpy_points[t,0]], [numpy_points[s,1], numpy_points[t,1]],[numpy_points[s,2], numpy_points[t,2]], c=color)

def test_draw():
  import numpy as np
  from pyquaternion import Quaternion
  parts = []
  for i in range(5):
    quat = Quaternion.random()
    trans = np.random.rand(3) - 0.5
    shape = np.clip(np.random.rand(3), a_min=0.05, a_max=0.3)
    part = Variable(torch.Tensor(
      [shape[0], shape[1], shape[2], trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]])).view(1, 10)
    parts.append(part)
  parts = torch.cat(parts)
  plot_parts(parts)

def test_plot3():
  import numpy as np
  x = torch.from_numpy(np.random.rand(100,3))
  plot3(x)

if __name__  == "__main__":
  test_plot3()



