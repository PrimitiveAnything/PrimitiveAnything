from skimage import measure



def march(tsdf):
  verts, faces, normals, values = measure.marching_cubes_lewiner(tsdf, 0)
  trianges = verts[faces]
  return trianges


def writeObj(filename, triangles):
  fout = open(filename, 'w')
  nv = 1
  for t in range(triangles.shape[0]):
    for p in range(3):
      fout.write('v {} {} {}\n'.format(triangles[t][p][0], triangles[t][p][1], triangles[t][p][2]))

    fout.write('f {} {} {}\n'.format(nv, nv+1, nv+2))
    nv = nv + 3
  fout.close()


