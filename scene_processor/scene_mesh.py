import os
import math
import numpy as np
import trimesh
import trimesh.visual
from typing import Dict
import random
try:
    from scene_config import SceneConfig, MaterialConfig
    from remesh import remesh, robust_remesh
except ImportError:
    from renderformer.renderformer.scene_processor.scene_config import SceneConfig, MaterialConfig
    from renderformer.renderformer.scene_processor.remesh import remesh, robust_remesh


def safe_remesh(vertices, faces, target_face_num, tolerance=1.5):
    """Remesh with a guard against pymeshlab blow-ups.

    pymeshlab's isotropic explicit remeshing occasionally returns vertices thousands of
    units away from its input (seen on ~0.5 % of Objaverse objects, all non-watertight,
    deterministic per mesh). The renderer then draws the object as giant shards and the
    dataset stores the same garbage, in ~5 % of scenes. Any result whose vertices leave
    a sphere of `tolerance` x the input's own radius (about its centroid) is rejected;
    the clustering-based robust_remesh is tried next; if that fails too, the input is
    returned unchanged (more faces than the target, but correct geometry).

    Returns (vertices, faces, method) with method in {'remesh', 'robust_remesh', 'none'}.
    Both the renderer and the h5 converter must call this, so that they stay identical.
    """
    vertices = np.asarray(vertices, dtype=np.float64); faces = np.asarray(faces)
    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center, axis=1).max()
    limit = tolerance * radius + 1e-6
    for fn in (remesh, robust_remesh):
        try:
            v, f = fn(vertices, faces, target_face_num)
        except Exception:
            continue
        v = np.asarray(v); f = np.asarray(f)
        if len(f) == 0 or not np.isfinite(v).all():
            continue
        if np.linalg.norm(v - center, axis=1).max() <= limit:
            return v, f, fn.__name__
    return vertices, faces, 'none'

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


def generate_scene_mesh(scene_config: SceneConfig, output_path: str, scene_config_dir: str,
                        geom_cache_path: str = None) -> None:
    """Generate combined mesh from scene configuration using trimesh.Scene

    With geom_cache_path set, each object's post-normalize/post-remesh geometry is
    cached there along with its object-to-world matrix, and the h5 converter reuses
    it instead of repeating trimesh.load + pymeshlab remesh. That second pass is the
    single most expensive step in the pipeline and produces, by construction, exactly
    what this function already computed.
    """
    split_mesh_folder_path = os.path.dirname(output_path) + '/split'
    os.makedirs(split_mesh_folder_path, exist_ok=True)
    geom_cache = {} if geom_cache_path else None

    for obj_key, obj_config in scene_config.objects.items():
        # if obj_config.mesh_path.endswith(".glb") or ("shapenet" in obj_config.mesh_path.lower()):
        #     mesh: trimesh.Trimesh = trimesh.load(scene_config_dir + '/' + obj_config.mesh_path, process=False, force='mesh')  # type: ignore
        # else:
        #     mesh: trimesh.Trimesh = trimesh.load(scene_config_dir + '/' + obj_config.mesh_path, process=False)  # type: ignore

        mesh_path = obj_config.mesh_path
        if os.path.isabs(mesh_path):
            final_mesh_path = mesh_path
        elif mesh_path.startswith('~'):
            final_mesh_path = os.path.expanduser(mesh_path)
        else:
            final_mesh_path = os.path.normpath(os.path.join(scene_config_dir, mesh_path))
            
        kwargs = {'process': False}
        if mesh_path.endswith('.glb') or ('shapenet' in mesh_path.lower()):
            kwargs['force'] = 'mesh'
            
        mesh: trimesh.Trimesh = trimesh.load(final_mesh_path, **kwargs)
        if obj_config.transform.normalize:
            mesh = normalize_to_unit_sphere(mesh)
        # Remesh if existing is greater than target number
        if obj_config.remesh and mesh.faces.shape[0] > obj_config.remesh_target_face_num:
            new_v, new_f, how = safe_remesh(mesh.vertices, mesh.faces, obj_config.remesh_target_face_num)
            print(f'remesh {obj_key} from {mesh.faces.shape[0]} to {new_f.shape[0]} via {how}')
            mesh = trimesh.Trimesh(
                vertices=new_v,
                faces=new_f,
                process=False
            )



        # Snapshot before shading/colouring: those steps re-lay-out vertices but keep
        # the same triangles, so this is the surface the renderer draws and the surface
        # the converter samples.
        if geom_cache is not None:
            cached_v = np.asarray(mesh.vertices, dtype=np.float64).copy()
            cached_f = np.asarray(mesh.faces, dtype=np.int64).copy()

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
        obj_to_world = np.eye(4)
        for axis, angle in enumerate(transform.rotation):
            axis_array = np.array([1, 0, 0] if axis == 0 else [0, 1, 0] if axis == 1 else [0, 0, 1]).astype(float)
            rotation_matrix = trimesh.transformations.rotation_matrix(
                np.deg2rad(angle), axis_array
            )
            mesh.apply_transform(rotation_matrix)
            obj_to_world = rotation_matrix @ obj_to_world

        mesh.apply_scale(transform.scale)
        mesh.apply_translation(transform.translation)
        # The same three operations as one matrix. This is exactly what
        # utils.get_transform_matrix(..., apply_normalize=False) rebuilds from the JSON;
        # recording it here means the converter never has to rebuild it.
        scale_vec = transform.scale if hasattr(transform.scale, '__len__') else [transform.scale] * 3
        obj_to_world = np.diag([scale_vec[0], scale_vec[1], scale_vec[2], 1.0]) @ obj_to_world
        obj_to_world = trimesh.transformations.translation_matrix(transform.translation) @ obj_to_world

        if geom_cache is not None:
            geom_cache[obj_key] = (cached_v, cached_f, obj_to_world)

        print(f'object {obj_key} vertex normals:', mesh.vertex_normals.shape)  # must have this line to trigger the calculation of vertex normals

        # Save individual meshes
        mesh.export(f"{split_mesh_folder_path}/{obj_key}.obj", include_normals=True, include_texture=True)

    if geom_cache is not None:
        # Written only once every object succeeded, then renamed into place, so a
        # converter racing the renderer never observes a partial cache.
        arrays = {'__keys__': np.array(list(geom_cache.keys()))}
        for i, (k, (v, f, m)) in enumerate(geom_cache.items()):
            arrays[f'v{i}'], arrays[f'f{i}'], arrays[f'M{i}'] = v, f, m
        tmp = geom_cache_path + '.tmp.npz'   # savez_compressed appends .npz otherwise
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, geom_cache_path)
