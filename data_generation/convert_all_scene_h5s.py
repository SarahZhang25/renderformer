import os
import glob
import subprocess
import argparse
import concurrent.futures

def main():
    parser = argparse.ArgumentParser(description="Convert all JSON scenes in a directory to h5 using convert_scene.py")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing scene JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted h5 outputs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for conversion")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all JSON files in the input directory
    scene_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    
    if not scene_files:
        print(f"No JSON files found in {args.input_dir}")
        return
        
    print(f"Found {len(scene_files)} scenes to convert.")
    
    def convert_scene(scene_file):
        scene_name = os.path.splitext(os.path.basename(scene_file))[0]
        # Save directly in the output_dir
        output_h5_path = os.path.join(args.output_dir, f"{scene_name}.h5")
        
        print(f"\n--- Converting {scene_file} ---")
        
        # Construct the command
        cmd = [
            "python3", "scene_processor/convert_scene.py",
            scene_file,
            "--output_h5_path", output_h5_path
        ]
            
        print("Running command:", " ".join(cmd))
        
        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error converting {scene_file}: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(convert_scene, scene_files))
            
    print("\nAll conversion tasks completed!")

if __name__ == "__main__":
    main()
