"""
Generate json scenes.

Example usage:
    python data_generation/generate_scenes.py --num_scenes 10 --output_dir datasets/json_scenes/cbox3_ten_objs
"""

import os
import json
import random
import glob
import math
import argparse
import trimesh
import numpy as np

def find_objaverse_objects():
    objaverse_dir = os.path.expanduser("/home/sazhang/.objaverse")
    print(f"Using objaverse_dir: {objaverse_dir}")
    # find all glb files
    glb_files = []
    for root, dirs, files in os.walk(objaverse_dir):
        for f in files:
            if f.endswith('.glb'):
                glb_files.append(os.path.join(root, f))
    return glb_files

def generate_scene(template_json, objaverse_objects, scene_idx, output_dir, num_views=4):
    with open(template_json, 'r') as f:
        scene = json.load(f)
        
    # Pre-load template walls/backgrounds for camera occlusion checking
    template_meshes = []
    for obj_key, obj_info in scene.get('objects', {}).items():
        if "mesh_path" in obj_info and "backgrounds/" in obj_info["mesh_path"]:
            # The mesh path might be relative to template or elsewhere. 
            # We know it's in datasets/templates/backgrounds.
            mesh_basename = os.path.basename(obj_info["mesh_path"])
            actual_mesh_path = os.path.join("datasets/templates/backgrounds", mesh_basename)
            if os.path.exists(actual_mesh_path):
                mesh = trimesh.load(actual_mesh_path, process=False, force='mesh')
                # apply transform
                transform = obj_info.get("transform", {})
                scale = transform.get("scale", [1.0, 1.0, 1.0])
                rotation = transform.get("rotation", [0.0, 0.0, 0.0])
                translation = transform.get("translation", [0.0, 0.0, 0.0])
                
                for axis, angle in enumerate(rotation):
                    axis_array = np.array([1, 0, 0] if axis == 0 else [0, 1, 0] if axis == 1 else [0, 0, 1]).astype(float)
                    rotation_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(angle), axis_array)
                    mesh.apply_transform(rotation_matrix)
                mesh.apply_scale(scale)
                mesh.apply_translation(translation)
                template_meshes.append(mesh)
    
    if template_meshes:
        scene_walls = trimesh.util.concatenate(template_meshes)
    else:
        scene_walls = None

        
    scene['scene_name'] = f"scene_{scene_idx}"
    
    # 1 to 3 random objects
    num_objects = random.randint(1, 3)
    selected_objects = random.sample(objaverse_objects, min(num_objects, len(objaverse_objects)))
    
    placed_bboxes = []
    
    actual_i = 0
    for obj_path in selected_objects:
        obj_key = f"objaverse_{actual_i}"
        
        # calculate relative path for mesh_path to avoid the prepended scene_config_dir issue
        rel_path = os.path.relpath(obj_path, output_dir)
        
        # Scale and rotation
        scale_val = random.uniform(0.3, 0.7)
        scale = [scale_val, scale_val, scale_val]
        rotation = [random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)]
        
        # Load mesh and replicate the exact transform pipeline from scene_mesh.py
        # to get the true vertex positions before translation is applied.
        # scene_mesh.py does: normalize -> remesh -> rotate -> scale -> translate
        # So final vertex positions = (transformed_vertices) + translation
        # We need: all (transformed_vertices + t) inside the box bounds.
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        
        # Guard: skip objects that loaded with no geometry
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print(f"  Warning: object {obj_key} loaded with no vertices, skipping ({obj_path})")
            continue
        
        # Step 1: normalize (same as scene_mesh.py normalize_to_unit_sphere)
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        bounding_sphere_radius = np.linalg.norm(mesh.vertices, ord=2, axis=-1).max() * 2.0
        mesh.vertices = mesh.vertices / bounding_sphere_radius
        
        # Step 2: rotation (same sequential axis rotation as scene_mesh.py)
        for axis, angle in enumerate(rotation):
            axis_array = np.array([1, 0, 0] if axis == 0 else [0, 1, 0] if axis == 1 else [0, 0, 1]).astype(float)
            rotation_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(angle), axis_array)
            mesh.apply_transform(rotation_matrix)
        
        # Step 3: scale
        mesh.apply_scale(scale)
        
        # Now get the actual min/max vertex coordinates (before translation)
        v_min = mesh.vertices.min(axis=0)  # [min_x, min_y, min_z]
        v_max = mesh.vertices.max(axis=0)  # [max_x, max_y, max_z]
        
        # Cornell box actual bounds (from wall geometry with scale=0.5):
        #   plane.obj (floor): Z = -0.5, X/Y in [-0.5, 0.5]
        #   wall0.obj (back):  Y = 0.5,  X/Z in [-0.5, 0.5]
        #   wall1.obj (right): X = 0.5,  Y/Z in [-0.5, 0.5]
        #   wall2.obj (left):  X = -0.5, Y/Z in [-0.5, 0.5]
        #   Open face at Y = -0.5, no ceiling
        # So interior: X in [-0.5, 0.5], Y in [-0.5, 0.5], Z in [-0.5, 0.5]
        box_lo = np.array([-0.5, -0.5, -0.5])
        box_hi = np.array([0.5, 0.5, 0.5])
        padding = 0.05
        
        t_lo = box_lo + padding - v_min  # minimum valid translation per axis
        t_hi = box_hi - padding - v_max  # maximum valid translation per axis
        
        # Check if object even fits
        if np.any(t_lo > t_hi):
            print(f"  Warning: object {obj_key} too large to fit in box, skipping")
            continue
        
        max_retries = 100
        placed = False
        
        for _ in range(max_retries):
            tx = random.uniform(t_lo[0], t_hi[0])
            ty = random.uniform(t_lo[1], t_hi[1])
            tz = random.uniform(t_lo[2], t_hi[2])
            
            # World-space AABB after translation
            b_min = v_min + np.array([tx, ty, tz])
            b_max = v_max + np.array([tx, ty, tz])
            
            # Check collision with already placed objects
            collision = False
            for pbox in placed_bboxes:
                if (b_min[0] < pbox[3] and b_max[0] > pbox[0] and
                    b_min[1] < pbox[4] and b_max[1] > pbox[1] and
                    b_min[2] < pbox[5] and b_max[2] > pbox[2]):
                    collision = True
                    break
            
            if not collision:
                translation = [tx, ty, tz]
                placed_bboxes.append((b_min[0], b_min[1], b_min[2], b_max[0], b_max[1], b_max[2]))
                placed = True
                break
                
        if not placed:
            continue
            
        actual_i += 1
        
        # "randomly assign material parameters either per-shading-group or per-triangle with a 1:1 ratio"
        # updated to allow procedural patterns as well
        random_diffuse_type = random.choice(["per-shading-group", "procedural"]) # "per-triangle", 
        # "diffuse albedo with max intensity per color channel set such that sum with monochromatic specular lies between 0.9 and 1.0"
        sum_target = random.uniform(0.9, 1.0)
        specular_val = random.uniform(0.01, 0.5)
        random_diffuse_max = sum_target - specular_val
        specular = [specular_val, specular_val, specular_val]
        

        procedural_pattern, procedural_frequency, procedural_color_a, procedural_color_b = None, None, None, None

        if random_diffuse_type == "per-shading-group":
            diffuse = [random.uniform(0, random_diffuse_max) for _ in range(3)]
            random_diffuse_max_val = 0.0
            rand_seed = None
        elif "procedural" in random_diffuse_type:
            diffuse = [0.0, 0.0, 0.0]
            random_diffuse_max_val = random_diffuse_max
            rand_seed = random.randint(0, 1000000)
            procedural_pattern = "sinusoidal" #random.choice(["sinusoidal", "checkerboard"]) # "sinusoidal" or "checkerboard"
            procedural_frequency = [random.uniform(5.0, 20.0) for _ in range(3)]
            procedural_color_a = [random.uniform(0, random_diffuse_max) for _ in range(3)]
            procedural_color_b = [random.uniform(0, random_diffuse_max) for _ in range(3)]
        else:
            diffuse = [0.0, 0.0, 0.0]
            random_diffuse_max_val = random_diffuse_max
            rand_seed = random.randint(0, 1000000)
            
        # Roughness: log-sampled in [0.01, 1.0]
        roughness = math.exp(random.uniform(math.log(0.01), math.log(1.0)))
        
        smooth_shading = random.random() < 0.5
        
        # remesh_target = random.randint(256, 3072) # TODO: this needs to be strategic to only apply for too high face count
        
        scene['objects'][obj_key] = {
            "mesh_path": rel_path,
            "transform": {
                "translation": translation,
                "rotation": rotation,
                "scale": scale,
                "normalize": True
            },
            "material": {
                "diffuse": diffuse,
                "specular": specular,
                "roughness": roughness,
                "emissive": [0.0, 0.0, 0.0],
                "smooth_shading": smooth_shading,
                "random_diffuse_max": random_diffuse_max_val,
                "random_diffuse_type": random_diffuse_type,
                "rand_tri_diffuse_seed": rand_seed,
                "procedural_pattern": procedural_pattern,
                "procedural_frequency": procedural_frequency,
                "procedural_color_a": procedural_color_a,
                "procedural_color_b": procedural_color_b
            },
            "remesh": True,
            "remesh_target_face_num": 512 #remesh_target
        }
        
    # Camera
    scene['cameras'] = []
    for _ in range(num_views):
        placed_camera = False
        for _ in range(100): # max retries for unoccluded camera
            # FOV uniformly sampled [30, 60]
            fov = random.uniform(30.0, 60.0)
            # Distance uniformly sampled between 1.5 and 2.0 units 
            dist = random.uniform(1.5, 2.0)
            # angle
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(math.pi / 6, math.pi / 2.5) # mostly from above, but not straight top
            cam_pos = [dist * math.cos(theta) * math.sin(phi), dist * math.sin(theta) * math.sin(phi), dist * math.cos(phi)]
            look_at = [random.uniform(-0.2, 0.2) for _ in range(3)]
            
            occluded = False
            if scene_walls is not None:
                ray_origins = np.array([cam_pos])
                ray_directions = np.array([np.array(look_at) - np.array(cam_pos)])
                # Normalize direction
                ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True)
                
                # Check intersection
                intersections = scene_walls.ray.intersects_any(ray_origins, ray_directions)
                if intersections[0]:
                    occluded = True
            
            if not occluded:
                scene['cameras'].append({
                    "position": cam_pos,
                    "look_at": look_at,
                    "up": [0.0, 0.0, 1.0],
                    "fov": fov
                })
                placed_camera = True
                break
        
        if not placed_camera:
            print(f"Warning: Could not place an unoccluded camera in scene {scene_idx}")
            # just use the last generated one
            scene['cameras'].append({
                "position": cam_pos,
                "look_at": look_at,
                "up": [0.0, 0.0, 1.0],
                "fov": fov
            })

    
    # Lighting (1 to 8 light sources)
    keys_to_delete = [k for k in scene['objects'].keys() if k.startswith('light_')]
    for k in keys_to_delete:
        del scene['objects'][k]
        
    num_lights = random.randint(1, 8)
    for i in range(num_lights):
        # Distance [2.1, 2.7]
        l_dist = random.uniform(2.1, 2.7)
        l_theta = random.uniform(0, 2*math.pi)
        l_phi = random.uniform(0, math.pi/6) # control azimuth to be mostly from the top, not too much from the sides to avoid wall/floor blocking. Can increase later for more diversity.
        l_pos = [l_dist * math.cos(l_theta) * math.sin(l_phi), l_dist * math.sin(l_theta) * math.sin(l_phi), l_dist * math.cos(l_phi)]
        
        # Intensity [2500, 5000]
        intensity = random.uniform(2500, 5000)
        
        scene['objects'][f"light_{i}"] = {
            "mesh_path": "../../templates/lighting/tri.obj",
            "transform": {
                "translation": l_pos,
                "rotation": [0.0, math.degrees(l_phi), math.degrees(l_theta)],
                "scale": [2.5, 2.5, 2.5],
                "normalize": False
            },
            "material": {
                "diffuse": [1.0, 1.0, 1.0],
                "specular": [0.0, 0.0, 0.0],
                "roughness": 1.0,
                "emissive": [intensity, intensity, intensity],
                "smooth_shading": False,
                "random_diffuse_max": 0.0,
                "random_diffuse_type": "per-shading-group",
                "rand_tri_diffuse_seed": None
            },
            "remesh": False
        }

    # fix relative paths for background objects from the template
    for k, v in scene['objects'].items():
        if v["mesh_path"].startswith("backgrounds/"):
            v["mesh_path"] = "../../templates/" + v["mesh_path"]
            
    with open(os.path.join(output_dir, f"scene_{scene_idx:04d}.json"), 'w') as f:
        json.dump(scene, f, indent=4)

if __name__ == "__main__":
    ## set n_scenes as args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="datasets/json_scenes/new_dataset")
    parser.add_argument("--num_views", type=int, default=4)
    args = parser.parse_args()
    num_scenes = args.num_scenes
    output_dir = args.output_dir

    random.seed(42) # fix seed
    
    # We will run this script from the renderformer directory
    os.makedirs(output_dir, exist_ok=True)
    template_jsons = [f"datasets/templates/cbox-{i}-walls.json" for i in range(0, 4)]
    
    print("Finding objaverse objects...")
    objaverse_objects = find_objaverse_objects()
    # objaverse_objects = random.sample(objaverse_objects, 5) # restrict to N objs
    print(f"Found {len(objaverse_objects)} glb files.")
    
    if len(objaverse_objects) == 0:
        print("No objaverse objects found. Please ensure ~/.objaverse contains .glb files.")
        
    print(f"Generating {num_scenes} scenes...")
    for i in range(num_scenes):
        template_json = random.choice(template_jsons)
        generate_scene(template_json, objaverse_objects, i, output_dir, num_views=args.num_views)
    print(f"Done! Generated {num_scenes} JSON scenes in {output_dir}")
