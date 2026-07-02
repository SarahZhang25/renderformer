import os
import math
import numpy as np
import trimesh
import trimesh.visual
from typing import Dict
import random
from scene_config import SceneConfig, MaterialConfig
from remesh import remesh

def get_procedural_color(vertices: np.ndarray, config: MaterialConfig) -> np.ndarray:
    pattern = config.procedural_pattern
    freq = np.array(config.procedural_frequency) if config.procedural_frequency else np.array([10.0, 10.0, 10.0])
    color_a = np.array(config.procedural_color_a) if config.procedural_color_a else np.array([1.0, 0.0, 0.0])
    color_b = np.array(config.procedural_color_b) if config.procedural_color_b else np.array([0.0, 0.0, 1.0])
    
    if pattern == "sinusoidal":
        val = np.sin(vertices[:, 0] * freq[0]) + np.sin(vertices[:, 1] * freq[1]) + np.sin(vertices[:, 2] * freq[2])
        weight = (val + 3.0) / 6.0
    elif pattern == "checkerboard":
        val = np.floor(vertices[:, 0] * freq[0]) + np.floor(vertices[:, 1] * freq[1]) + np.floor(vertices[:, 2] * freq[2])
        weight = val % 2
    else:
        weight = np.random.rand(vertices.shape[0])
        
    weight = weight[:, None]
    colors = color_a * weight + color_b * (1.0 - weight)
    return colors

def normalize_to_unit_sphere(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to fit in a unit sphere centered at origin"""
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    bounding_sphere_radius = np.linalg.norm(mesh.vertices, ord=2, axis=-1).max() * 2.
    mesh.vertices = mesh.vertices / bounding_sphere_radius

    return mesh


def generate_scene_mesh(scene_config: SceneConfig, output_path: str, scene_config_dir: str) -> None:
    """Generate combined mesh from scene configuration using trimesh.Scene"""
    split_mesh_folder_path = os.path.dirname(output_path) + '/split'
    os.makedirs(split_mesh_folder_path, exist_ok=True)

    for obj_key, obj_config in scene_config.objects.items():
        if obj_config.mesh_path.endswith(".glb") or ("shapenet" in obj_config.mesh_path.lower()):
            mesh: trimesh.Trimesh = trimesh.load(scene_config_dir + '/' + obj_config.mesh_path, process=False, force='mesh')  # type: ignore
        else:
            mesh: trimesh.Trimesh = trimesh.load(scene_config_dir + '/' + obj_config.mesh_path, process=False)  # type: ignore
        if obj_config.transform.normalize:
            mesh = normalize_to_unit_sphere(mesh)
        if obj_config.remesh:
            new_v, new_f = remesh(mesh.vertices, mesh.faces, obj_config.remesh_target_face_num)
            print(f'remesh {obj_key} from {mesh.faces.shape[0]} to {new_f.shape[0]}')
            mesh = trimesh.Trimesh(
                vertices=new_v,
                faces=new_f,
                process=False
            )



        if obj_config.material.smooth_shading:
            mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(30))
        else:  # ordinary vertex color
            mesh = trimesh.Trimesh(
                vertices=mesh.triangles.reshape(-1, 3),
                faces=np.arange(len(mesh.triangles) * 3).reshape(-1, 3),
                process=False
            )

        if obj_config.material.rand_tri_diffuse_seed is not None or obj_config.material.random_diffuse_type == "procedural":
            # to make the random color consistent
            seed = obj_config.material.rand_tri_diffuse_seed if obj_config.material.rand_tri_diffuse_seed is not None else 42
            random.seed(seed)
            np.random.seed(seed)

            # apply diffuse properties
            mesh_split = []
            kwarg: Dict = {'only_watertight': False}
            if obj_config.material.random_diffuse_type in ["per-triangle", "procedural"]:
                kwarg['adjacency'] = np.array([])  # force split by triangle
            for small_mesh in mesh.split(**kwarg):
                if obj_config.material.random_diffuse_type == "procedural":
                    centroid = small_mesh.vertices.mean(axis=0, keepdims=True)
                    color_val = get_procedural_color(centroid, obj_config.material)
                    shared_color = (color_val * 255).clip(0, 255).astype(int).repeat(small_mesh.faces.shape[0], axis=0)
                elif obj_config.material.random_diffuse_type == "per-triangle":
                    shared_color = np.random.randint(0, math.ceil(256 * obj_config.material.random_diffuse_max), (1, 3)).repeat(small_mesh.faces.shape[0], axis=0)
                else:
                    raise ValueError(f"Unknown random_diffuse_type: {obj_config.material.random_diffuse_type}")
                new_small_mesh = trimesh.Trimesh(
                    vertices=small_mesh.vertices,
                    faces=small_mesh.faces,
                    vertex_normals=small_mesh.vertex_normals,
                    face_colors=shared_color,
                    process=False
                )
                mesh_split.append(new_small_mesh)
            mesh = trimesh.util.concatenate(mesh_split)
        else:
            vertex_colors = (np.array(obj_config.material.diffuse) * 255.).clip(0, 255).astype(int)
            vertex_colors = np.tile(vertex_colors, mesh.vertices.shape[0]).reshape(-1, 3)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=vertex_colors,
            )

        # Apply transformations (after texturing so textures evaluate in local object coordinates)
        transform = obj_config.transform

        # first apply rotation, then scale, then translation
        for axis, angle in enumerate(transform.rotation):
            axis_array = np.array([1, 0, 0] if axis == 0 else [0, 1, 0] if axis == 1 else [0, 0, 1]).astype(float)
            rotation_matrix = trimesh.transformations.rotation_matrix(
                np.deg2rad(angle), axis_array
            )
            mesh.apply_transform(rotation_matrix)
        
        mesh.apply_scale(transform.scale)
        mesh.apply_translation(transform.translation)

        print(f'object {obj_key} vertex normals:', mesh.vertex_normals.shape)  # must have this line to trigger the calculation of vertex normals

        # Save individual meshes
        mesh.export(f"{split_mesh_folder_path}/{obj_key}.obj", include_normals=True, include_texture=True)
