import torch
import math
import numpy as np
import trimesh

from collections import OrderedDict

def GetMatrixList(mesh, voxel_size):
	lower_bounds = mesh.bounds[0]
	upper_bounds = mesh.bounds[1]
	
	x_steps = int((upper_bounds[0] - lower_bounds[0]) / voxel_size)
	y_steps = int((upper_bounds[1] - lower_bounds[1]) / voxel_size)
	z_steps = int((upper_bounds[2] - lower_bounds[2]) / voxel_size)
	
	x = torch.linspace(lower_bounds[0], upper_bounds[0], x_steps + 1)
	y = torch.linspace(lower_bounds[1], upper_bounds[1], y_steps + 1)
	z = torch.linspace(lower_bounds[2], upper_bounds[2], z_steps + 1)
	
	matrix_list = torch.meshgrid([x, y, z], indexing = "xy")
	matrix_list = torch.stack(matrix_list)
	matrix_list = matrix_list.reshape(3, -1).T
	
	print("    [Matrix Size] :", matrix_list.shape[0])
	print("    [Lower Bounds]:", lower_bounds)
	print("    [Upper Bounds]:", upper_bounds)
	
	return matrix_list

mesh_path = "ranch_mesh.glb"
point_path = "ranch.ply"

scene = trimesh.load(mesh_path, force = "scene", group_material = False)
mesh_list = scene.geometry

if not isinstance(mesh_list, OrderedDict):
	meshes_dict = OrderedDict()
	meshes_dict["mesh"] = mesh_list
	mesh_list = meshes_dict

particle_data_list = []

iterations = 0
for mesh_name in mesh_list:
	completion = int(100 * iterations / len(mesh_list))
	iterations += 1
	print(str(completion) + "% -", mesh_name)
	
	mesh = mesh_list[mesh_name]
	matrix_list = GetMatrixList(mesh, 0.4)
	
	is_contained = mesh.contains(matrix_list)
	particle_points = matrix_list[is_contained].cuda()
	
	if particle_points.shape[0] < 1:
		print("    <Warning>:", mesh_name, "has particle count of 0. Ignoring mesh.")
		continue
		
	particle_data_list.append(particle_points)
	 
vertices = torch.cat(particle_data_list, dim = 0).cpu()
point_mesh = trimesh.Trimesh(vertices = vertices)

print("[Total Particles] :", vertices.shape[0])

point_mesh.export(point_path)