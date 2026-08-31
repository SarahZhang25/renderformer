# NOTE: install bpy (4.0.0) and bpy_helper (0.0.8) before usage
# pip install bpy==4.0.0 bpy_helper==0.0.8 --extra-index-url https://download.blender.org/pypi/

import bpy
from bpy_helper.camera import create_camera, look_at_to_c2w
from bpy_helper.material import create_specular_roughness_material, create_white_emmissive_material
from bpy_helper.scene import reset_scene, import_3d_model, scene_meshes
from bpy_helper.utils import stdout_redirected
from bpy_helper.io import save_blend_file

import os
import json
import tempfile
import imageio
import numpy as np
from tqdm import tqdm
from dacite import from_dict, Config

BLENDER_BACKEND = os.getenv('BLENDER_BACKEND', 'CUDA')

from scene_config import SceneConfig, CameraConfig
from scene_mesh import generate_scene_mesh


def scene_to_img(
        scene_config: SceneConfig,
        mesh_path: str,
        output_image_path: str,
        save_img: bool = False,
        resolution: int = 512,
        spp: int = 4096,
        dump_blend_file: bool = True,
        skip_rendering: bool = True
    ) -> list[tuple[np.ndarray, np.ndarray]]:


    def setup_blender_scene(scene_config: SceneConfig, mesh_path: str) -> None:
        reset_scene()
        with stdout_redirected():
            split_mesh_path = os.path.dirname(mesh_path) + '/split'
            
            for obj_key, obj_config in scene_config.objects.items():
                import_3d_model(f'{split_mesh_path}/{obj_key}.obj')
                material_config = obj_config.material
                emissive = obj_config.material.emissive
                if any(e > 0 for e in emissive):
                    max_val = max(emissive)
                    material = create_white_emmissive_material(
                        strength=max_val,
                        material_name=f"{obj_key}"
                    )
                    # Normalize RGB color tint and apply to BSDF node (Blender 4.0+)
                    color = (emissive[0] / max_val, emissive[1] / max_val, emissive[2] / max_val, 1.0)
                    bsdf = material.node_tree.nodes["Principled BSDF"]
                    bsdf.inputs["Emission Color"].default_value = color
                    bsdf.inputs["Base Color"].default_value = color
                else:
                    material = create_specular_roughness_material(
                        diffuse_color=tuple(material_config.diffuse),
                        specular_color=tuple(material_config.specular),
                        roughness=material_config.roughness,
                        material_name=f"{obj_key}"
                    )
                    if material_config.rand_tri_diffuse_seed is not None or material_config.random_diffuse_type == "procedural":  # Use vertex color as diffuse color when have per-triangle diffuse color
                        bsdf = material.node_tree.nodes["Group"]
                        vcol = material.node_tree.nodes.new(type="ShaderNodeVertexColor")
                        material.node_tree.links.new(vcol.outputs['Color'], bsdf.inputs['Diffuse'])
                
                for obj in scene_meshes():
                    if obj.name == obj_key:
                        obj.data.materials.clear()  # Clear all materials
                        obj.data.materials.append(material)
                        obj.rotation_mode = 'XYZ'
                        obj.rotation_euler = (0.0, 0.0, 0.0)  # Set rotation to (0, 0, 0)

    def render_scene(camera_config: CameraConfig, output_image_path: str, output_obj_path: str) -> tuple[np.ndarray, np.ndarray]:
        camera_pos = np.array(camera_config.position)
        look_at = np.array(camera_config.look_at)
        up = np.array(camera_config.up)
        fov = camera_config.fov
        
        c2w = look_at_to_c2w(camera_pos, look_at, up)
        
        temp_img_path = output_image_path.replace(".png", ".exr")
        exr_valid = os.path.exists(temp_img_path) and os.path.getsize(temp_img_path) >= 1024
        png_valid = os.path.exists(output_image_path) and os.path.getsize(output_image_path) > 0
        
        if exr_valid:
            if save_img and not png_valid:
                # EXR exists but PNG is missing. Skip Blender, just generate the PNG.
                print(f"Skipping render for {output_image_path} (EXR exists), but generating missing PNG.", flush=True)
                img = imageio.v3.imread(temp_img_path).copy()
                imageio.v3.imwrite(output_image_path, (img * 255).clip(0, 255).astype(np.uint8))
            else:
                print(f"Skipping render for {output_image_path}, valid files already exist.", flush=True)
            return np.zeros((resolution, resolution, 4), dtype=np.float32), c2w
        
        if skip_rendering:
            return np.zeros((resolution, resolution, 4), dtype=np.float32), c2w

        camera = create_camera(c2w, fov)
        bpy.context.scene.camera = camera
        
        bpy.context.scene.render.resolution_x = resolution
        bpy.context.scene.render.resolution_y = resolution
        bpy.context.scene.render.engine = 'CYCLES'
        # Fleet mode: many worker processes share one node - cap Blender's
        # internal thread pool or each worker spawns ~cores threads (386 seen)
        # and 128 workers melt the box (BLENDER_THREADS env, default 2).
        bpy.context.scene.render.threads_mode = 'FIXED'
        bpy.context.scene.render.threads = int(os.environ.get('BLENDER_THREADS', '2'))
        bpy.context.scene.cycles.samples = spp
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        
        cycles_pref = bpy.context.preferences.addons["cycles"].preferences
        cycles_pref.compute_device_type = BLENDER_BACKEND
        cycles_pref.get_devices()
        for device in cycles_pref.devices:
            if device.type == BLENDER_BACKEND or (BLENDER_BACKEND == 'OPTIX' and device.type == 'CUDA'):
                device.use = True
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.threads = 8
        bpy.context.scene.render.threads_mode = 'FIXED'

        bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.  # remove all ambient

        if True:
            temp_img_path = output_image_path.replace(".png", ".exr")
            bpy.context.scene.render.filepath = os.path.abspath(temp_img_path)
            
            max_retries = 3
            for attempt in range(max_retries):
                bpy.ops.render.render(animation=False, write_still=True)
                
                # Safety check: if Blender hits VRAM OOM or output dir is full, it silently saves a 0-byte or truncated file
                if os.path.exists(temp_img_path) and os.path.getsize(temp_img_path) >= 1024:
                    print(f"Successfully rendered {temp_img_path}")
                    break
                    
                import time
                if attempt < max_retries - 1:
                    time.sleep(30)
                else:
                    file_size = os.path.getsize(temp_img_path) if os.path.exists(temp_img_path) else 0
                    raise RuntimeError(
                        f"Blender failed to render {temp_img_path} properly (file size is {file_size} bytes) after {max_retries} attempts. "
                        f"This usually means the GPU ran out of VRAM (try lowering --workers_per_gpu) or the output dir is full."
                    )
                
            img = imageio.v3.imread(temp_img_path, plugin="EXR-FI").copy()
            print(f"Loaded image from {temp_img_path} using EXR-FI plugin, shape: {img.shape}")
            if img.shape[0] == 0:
                raise RuntimeError(f"imageio failed to read the EXR file (read 0 frames). The file is likely corrupted.")
                
            if save_img:
                imageio.v3.imwrite(output_image_path, (img * 255).clip(0, 255).astype(np.uint8))

        return img, c2w

    setup_blender_scene(scene_config, mesh_path)

    results = []
    for i, camera_config in tqdm(list(enumerate(scene_config.cameras))):
        img, c2w = render_scene(camera_config, f"{output_image_path}_{i}.png", f"{output_image_path}_{i}.obj")
        results.append((img, c2w))

    # dump if needed
    if dump_blend_file:
        bpy.ops.file.pack_all()
        save_blend_file(output_image_path + '.blend')

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Render scenes using Blender')
    parser.add_argument('scene_config', type=str, help='Path to scene config JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for rendered images')
    parser.add_argument('--mesh_path', type=str, 
                       help='Path to mesh file. If not provided, a temporary directory will be used',
                       default=None)
    parser.add_argument('--no_dump_blend', dest='dump_blend', action='store_false', help='Do not save Blender file after rendering')
    parser.add_argument('--save_img', default=False, action='store_true', help='Save rendered images')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the rendered images')
    parser.add_argument('--spp', type=int, default=4096, help='Samples per pixel')
    
    args = parser.parse_args()
    
    with open(args.scene_config) as f:
        scene_config = json.load(f)
    scene_config = from_dict(data_class=SceneConfig, data=scene_config, config=Config(check_types=True, strict=True))
        
    os.makedirs(args.output_dir, exist_ok=True)
    output_base = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.scene_config))[0])
    print(f"Output base path: {output_base}")
    
    if args.mesh_path is None:
        print("No mesh path provided, using temporary directory")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
            print(f"Generating mesh in temporary path: {temp_mesh_path}")
            generate_scene_mesh(scene_config, temp_mesh_path, os.path.dirname(args.scene_config))
            scene_to_img(
                scene_config=scene_config,
                mesh_path=temp_mesh_path,
                output_image_path=output_base,
                dump_blend_file=args.dump_blend,
                save_img=args.save_img,
                resolution=args.resolution,
                spp=args.spp,
                skip_rendering=False
            )
    else:
        print(f"Using provided mesh path: {args.mesh_path}")
        generate_scene_mesh(scene_config, args.mesh_path, os.path.dirname(args.scene_config))
        scene_to_img(
            scene_config=scene_config,
            mesh_path=args.mesh_path,
            output_image_path=output_base,
            dump_blend_file=args.dump_blend,
            save_img=args.save_img,
            resolution=args.resolution,
            spp=args.spp,
            skip_rendering=False
        )
