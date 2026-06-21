"""
python3 data_generation/render_all_scenes.py --input_dir <input_dir> --output_dir <output_dir> --num_workers 4
"""

import os
import glob
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Render all JSON scenes in a directory using to_blend.py")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing scene JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the rendered outputs")
    parser.add_argument("--spp", type=int, default=1024, help="Samples per pixel for rendering")
    parser.add_argument("--save_img", action="store_true", default=True, help="Save rendered images")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for rendering")
    parser.add_argument("--no_dump_blend", action="store_true", default=True, help="Do not save Blender file after rendering")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU indices to use (e.g., 0,1,2,3). If not specified, auto-detects all available GPUs.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all JSON files in the input directory
    scene_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    
    if not scene_files:
        print(f"No JSON files found in {args.input_dir}")
        return
        
    print(f"Found {len(scene_files)} scenes to render.")
    
    # 1. Determine available GPUs
    gpu_list = []
    if args.gpus:
        try:
            gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        except ValueError:
            print("Invalid format for --gpus. Please use comma-separated integers (e.g. 0,1,2).")
            return
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        env_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        if env_gpus:
            try:
                gpu_list = [int(x.strip()) for x in env_gpus.split(",") if x.strip()]
            except ValueError:
                pass
                
    if not gpu_list:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_gpus))
        except ImportError:
            # Fallback: check nvidia-smi
            try:
                res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
                lines = [line for line in res.stdout.split("\n") if line.strip()]
                gpu_list = list(range(len(lines)))
            except Exception:
                gpu_list = []
                
    if gpu_list:
        print(f"Assigning rendering tasks round-robin to visible GPUs: {gpu_list}")
    else:
        print("No GPUs detected. Running without explicit GPU affinity.")
        
    import threading
    gpu_counter = 0
    gpu_lock = threading.Lock()
    
    def render_scene(scene_file):
        gpu_id = None
        if gpu_list:
            with gpu_lock:
                nonlocal gpu_counter
                gpu_id = gpu_list[gpu_counter % len(gpu_list)]
                gpu_counter += 1
                
        if gpu_id is not None:
            print(f"\n--- Rendering {scene_file} on GPU {gpu_id} ---")
        else:
            print(f"\n--- Rendering {scene_file} ---")
        
        # Construct the command
        cmd = [
            "python3", "scene_processor/to_blend.py",
            scene_file,
            "--output_dir", args.output_dir,
            "--spp", str(args.spp)
        ]
        
        if args.save_img:
            cmd.append("--save_img")
            
        if args.no_dump_blend:
            cmd.append("--no_dump_blend")
            
        print("Running command:", " ".join(cmd))
        
        # Configure environment with CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
        # Execute the command
        try:
            subprocess.run(cmd, check=True, env=env)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error rendering {scene_file}: {e}")
            return False

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(render_scene, scene_files))
            
    print("\nAll rendering tasks completed!")

if __name__ == "__main__":
    main()
