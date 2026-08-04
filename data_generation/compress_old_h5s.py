import os
import h5py
import glob
import argparse
from tqdm import tqdm
import concurrent.futures

def compress_h5(file_path):
    temp_path = file_path + ".tmp"
    
    with h5py.File(file_path, 'r') as f_in:
        # Some dataloaders group scenes into chunks, others are single scenes per file.
        # Let's handle both.
        is_chunked = False
        if 'triangles' not in f_in.keys():
            # It's a chunked file, iterate over scenes
            is_chunked = True
            
        needs_compression = False
        
        if is_chunked:
            for scene_name in f_in.keys():
                if len(f_in[scene_name]['texture'].shape) == 4:
                    needs_compression = True
                    break
        else:
            if len(f_in['texture'].shape) == 4:
                needs_compression = True
                
        if not needs_compression:
            return (False, 0, 0)
            
        with h5py.File(temp_path, 'w') as f_out:
            if is_chunked:
                scene_keys = list(f_in.keys())
                # Added inner tqdm per-scene
                for scene_name in tqdm(scene_keys, desc=f"Scenes in {os.path.basename(file_path)}", leave=False):
                    grp_out = f_out.create_group(scene_name)
                    grp_in = f_in[scene_name]
                    for key in grp_in.keys():
                        if key == 'texture' and len(grp_in[key].shape) == 4:
                            slim_texture = grp_in['texture'][:, :, 0, 0]
                            grp_out.create_dataset("texture", data=slim_texture, compression="gzip", compression_opts=9)
                        else:
                            grp_out.create_dataset(key, data=grp_in[key][:], compression="gzip", compression_opts=9)
            else:
                for key in f_in.keys():
                    if key == 'texture' and len(f_in[key].shape) == 4:
                        slim_texture = f_in['texture'][:, :, 0, 0]
                        f_out.create_dataset("texture", data=slim_texture, compression="gzip", compression_opts=9)
                    else:
                        f_out.create_dataset(key, data=f_in[key][:], compression="gzip", compression_opts=9)
                    
    # Backup the old bloated file and rename the temp file to the original name
    old_format_path = file_path.replace(".h5", "_oldformat.h5")
    os.rename(file_path, old_format_path)
    os.rename(temp_path, file_path)
    
    old_size = os.path.getsize(old_format_path)
    new_size = os.path.getsize(file_path)
    
    return (True, old_size, new_size)

def main():
    parser = argparse.ArgumentParser(description="Compress old bloated H5 files into the new lightweight format.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the .h5 files to compress")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()
    
    h5_files = glob.glob(os.path.join(args.dir, "rf_*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {args.dir}")
        return
        
    print(f"Found {len(h5_files)} .h5 files. Checking for bloated textures using {args.workers} workers...")
    
    compressed_count = 0
    total_old_size = 0
    total_new_size = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compress_h5, fp): fp for fp in h5_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(h5_files), desc="Files"):
            result = future.result()
            if result[0]:
                _, old_size, new_size = result
                compressed_count += 1
                total_old_size += old_size
                total_new_size += new_size
            
    print(f"\nDone! Compressed {compressed_count} bloated files.")
    if compressed_count > 0:
        old_mb = total_old_size / (1024 * 1024)
        new_mb = total_new_size / (1024 * 1024)
        saved_mb = old_mb - new_mb
        ratio = (new_mb / old_mb) * 100 if old_mb > 0 else 0
        print(f"Dataset has been compressed from {old_mb:.2f} MB down to {new_mb:.2f} MB (Saved {saved_mb:.2f} MB!)")
        print(f"New size is {ratio:.2f}% of the original size.")

if __name__ == "__main__":
    main()
