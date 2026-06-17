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
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all JSON files in the input directory
    scene_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    
    if not scene_files:
        print(f"No JSON files found in {args.input_dir}")
        return
        
    print(f"Found {len(scene_files)} scenes to render.")
    
    def render_scene(scene_file):
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
        
        # Execute the command
        try:
            subprocess.run(cmd, check=True)
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
