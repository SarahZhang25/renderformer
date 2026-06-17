import os
import json
import random
import glob
import math
import argparse
import trimesh
import numpy as np

def find_objaverse_objects():
    objaverse_dir = os.path.expanduser("~/.objaverse")
    # find all glb files
    glb_files = []
    for root, dirs, files in os.walk(objaverse_dir):
        for f in files:
            if f.endswith('.glb'):
                glb_files.append(os.path.join(root, f))
    return glb_files

def generate_scene(template_json, objaverse_objects, scene_idx, output_dir):
    with open(template_json, 'r') as f:
        scene = json.load(f)
        
    scene['scene_name'] = f"dataset0_scene_{scene_idx}"
    
    # 1 to 3 random objects
    num_objects = 1 #random.randint(1, 3)
    selected_objects = random.sample(objaverse_objects, min(num_objects, len(objaverse_objects)))
    
    placed_bboxes = []
    
    actual_i = 0
    for obj_path in selected_objects:
        obj_key = f"objaverse_{actual_i}"
        
        # calculate relative path for mesh_path to avoid the prepended scene_config_dir issue
        rel_path = os.path.relpath(obj_path, output_dir)
        
        # Scale and rotation
        scale_val = random.uniform(0.3, 0.5)
        scale = [scale_val, scale_val, scale_val]
        rotation = [random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)]
        
        # Load mesh and replicate the exact transform pipeline from scene_mesh.py
        # to get the true vertex positions before translation is applied.
        # scene_mesh.py does: normalize -> remesh -> rotate -> scale -> translate
        # So final vertex positions = (transformed_vertices) + translation
        # We need: all (transformed_vertices + t) inside the box bounds.
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        
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
        
        # Material
        # "randomly assign material parameters either per-shading-group or per-triangle with a 1:1 ratio"
        # fix to per-shading-group for now TODO: change later
        random_diffuse_type = "per-shading-group" #if random.random() < 0.5 else "per-triangle"
        # "diffuse albedo with max intensity per color channel set such that sum with monochromatic specular lies between 0.9 and 1.0"
        sum_target = random.uniform(0.9, 1.0)
        specular_val = random.uniform(0.01, 0.5)
        random_diffuse_max = sum_target - specular_val
        specular = [specular_val, specular_val, specular_val]
        
        if random_diffuse_type == "per-shading-group":
            diffuse = [random.uniform(0, random_diffuse_max) for _ in range(3)]
            random_diffuse_max_val = 0.0
            rand_seed = None
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
                "rand_tri_diffuse_seed": rand_seed
            },
            "remesh": True,
            "remesh_target_face_num": 512 #remesh_target
        }
        
    # Camera
    # FOV uniformly sampled [30, 60]
    fov = random.uniform(30.0, 60.0)
    # Distance uniformly sampled between 2.1 and 2.7 units 
    dist = random.uniform(2.1, 2.7)
    # angle
    # To avoid being blocked by the back/side walls or floor, place the camera in the front-top area
    # -Y is the open face of the box. So theta around 3*pi/2 (270 degrees)
    theta = random.uniform(1.25 * math.pi, 1.75 * math.pi)
    # phi from 60 to 90 degrees (pi/3 to pi/2) so it's slightly above or level, not below floor
    phi = random.uniform(math.pi / 3, math.pi / 2)
    cam_pos = [dist * math.cos(theta) * math.sin(phi), dist * math.sin(theta) * math.sin(phi), dist * math.cos(phi)]
    look_at = [random.uniform(-0.2, 0.2) for _ in range(3)]
    
    scene['cameras'] = [{
        "position": cam_pos,
        "look_at": look_at,
        "up": [0.0, 0.0, 1.0],
        "fov": fov
    }]
    
    # Lighting (1 to 8 light sources)
    keys_to_delete = [k for k in scene['objects'].keys() if k.startswith('light_')]
    for k in keys_to_delete:
        del scene['objects'][k]
        
    num_lights = random.randint(1, 1)# 8)
    for i in range(num_lights):
        # Distance [2.1, 2.7]
        l_dist = 2.1 #random.uniform(2.1, 2.7)
        l_theta = 0 #random.uniform(0, 2*math.pi)
        l_phi = 0 #random.uniform(0, math.pi/2)
        l_pos = [l_dist * math.cos(l_theta) * math.sin(l_phi), l_dist * math.sin(l_theta) * math.sin(l_phi), l_dist * math.cos(l_phi)]
        
        # Intensity [2500, 5000]
        intensity = 5000 # random.uniform(2500, 5000)
        
        scene['objects'][f"light_{i}"] = {
            "mesh_path": "../template_scenes/templates/lighting/tri.obj",
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
        if v["mesh_path"].startswith("templates/"):
            v["mesh_path"] = "../template_scenes/" + v["mesh_path"]
            
    with open(os.path.join(output_dir, f"scene_{scene_idx:04d}.json"), 'w') as f:
        json.dump(scene, f, indent=4)

if __name__ == "__main__":
    ## set n_scenes as args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="data_generation/dataset")
    args = parser.parse_args()
    num_scenes = args.num_scenes
    output_dir = args.output_dir

    random.seed(42) # fix seed
    
    # We will run this script from the renderformer directory
    os.makedirs(output_dir, exist_ok=True)
    template_json = "data_generation/template_scenes/cbox-3-walls.json"
    
    print("Finding objaverse objects...")
    objaverse_objects = find_objaverse_objects()
    objaverse_objects = random.sample(objaverse_objects, 1) # restrict to single obj
    print(f"Found {len(objaverse_objects)} glb files.")
    
    if len(objaverse_objects) == 0:
        print("No objaverse objects found. Please ensure ~/.objaverse contains .glb files.")
        
    print(f"Generating {num_scenes} scenes...")
    for i in range(num_scenes):
        generate_scene(template_json, objaverse_objects, i, output_dir)
    print(f"Done! Generated {num_scenes} JSON scenes in {output_dir}")
