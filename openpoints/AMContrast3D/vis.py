import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def vis_points(points, colors=None, labels=None, color_map='cubehelix_r', opacity=1.0, point_size=10.0):
    """Visualize a point cloud
    Note about direction in the visualization:  x: horizontal right (red arrow), y: vertical up (green arrow), and z: inside (blue arrow)
    Args:
        points ([np.array]): [N, 3] numpy array 
        colors ([type], optional): [description]. Defaults to None.
    """
    import pyvista as pv
    import numpy as np
    from pyvista import themes
    my_theme = themes.DefaultTheme()
    my_theme.color = 'black'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    pv.set_plot_theme(my_theme)

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()

    if colors is None and labels is not None:
        from matplotlib import cm
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        color_maps = cm.get_cmap(color_map)
        colors = color_maps(labels)

    plotter = pv.Plotter()
    plotter.add_points(points, opacity=opacity, point_size=point_size, render_points_as_spheres=True, scalars=colors, rgb=True)
    plotter.show()
            

def vis_obj(p, ambiguity, obj_name):
    coord = p.cpu().numpy().squeeze()
    value = ambiguity.cpu().numpy().squeeze()
    color = coord.view()
    for i in range(value.shape[0]):
        if value[i] == 0.0:
            color[i] = (218, 165, 32) ### Inner: Yellow
        elif value[i] == 1.0:
            color[i] = (139, 35, 35) ### Onlyone: Red
        elif value[i] == 0.5:
            color[i] = (34, 134, 34) ### Boundary: Green
    fout = open(obj_name, 'w')
    for i in range(coord.shape[0]):
        c = color[i] ### color[i,0] / 127.5-1, color[i,1] / 127.5-1, color[i,2] / 127.5-1
        fout.write('v %f %f %f %f %f %f\n' % (coord[i,0], coord[i,1], coord[i,2], color[i,0], color[i,1], color[i,2]))
    fout.close()


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def write_ply_point_normal(name, vertices, colors):
  fout = open(name, 'w')
  fout.write("ply\n")
  fout.write("format ascii 1.0\n")
  fout.write("element vertex "+str(len(vertices))+"\n")
  fout.write("property float x\n")
  fout.write("property float y\n")
  fout.write("property float z\n")
  fout.write("property uchar red\n")
  fout.write("property uchar green\n")
  fout.write("property uchar blue\n")
  fout.write("end_header\n")
  for ii in range(len(vertices)):
    fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(min(255,int(255*colors[ii,2])))+" "+str(min(255,int(255*colors[ii,1])))+" "+str(min(255,int(255*colors[ii,0])))+"\n")

