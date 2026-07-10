import objaverse
import os
import random
import time
import trimesh
import urllib.request
import urllib.error
import concurrent.futures
from tqdm import tqdm

random.seed(42)
MAX_VERTEX_COUNT = 50000

# Set base paths to your custom directory
objaverse.BASE_PATH = "/home/sazhang/.objaverse"
objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")
GLB_DOWNLOAD_DIR = "/dev/shm/objaverse"

def select_uids(num_objects=10, max_vertex_count=MAX_VERTEX_COUNT):
    print("Fetching LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()
    
    # Flatten all UIDs from LVIS annotations (using a set to ensure uniqueness)
    lvis_uids = set()
    for uids in lvis_annotations.values():
        lvis_uids.update(uids)
        
    lvis_uids = list(lvis_uids)
    print(f"Total unique LVIS-annotated objects available: {len(lvis_uids)}")
    
    # Shuffle to ensure we sample randomly across all categories
    random.shuffle(lvis_uids)
    
    # To find `num_objects` valid ones, we likely need to check more candidates than `num_objects`.
    # Let's inspect up to 5x the requested amount (capped at the total available).
    num_to_inspect = min(len(lvis_uids), num_objects * 5)
    subset_uids = lvis_uids[:num_to_inspect]
    
    print(f"Selected {len(subset_uids)} UIDs as candidates to inspect.\n")

    # 1. Inspecting the objects (Metadata)
    print("Loading annotations/metadata...")
    annotations = objaverse.load_annotations(uids=subset_uids)
    
    sample_uids = []
    n_skipped = 0
    scene_keywords = {'scene', 'interior', 'exterior', 'architecture', 'level', 'room'}

    for uid in subset_uids:
        if len(sample_uids) >= num_objects:
            break
            
        metadata = annotations.get(uid, {})
        
        # Extract details from metadata
        vertex_count = metadata.get('vertexCount', 0)
        face_count = metadata.get('faceCount', 0)
        animation_count = metadata.get('animationCount', 0)
        
        name = metadata.get('name', '').lower()
        tags = [tag.get('name', '').lower() for tag in metadata.get('tags', [])]
        categories = [cat.get('name', '').lower() for cat in metadata.get('categories', [])]
        
        # 1. Reject point clouds (no faces)
        if face_count == 0:
            n_skipped += 1
            continue
            
        # 2. Reject animations (optional, better for static scenes)
        if animation_count > 0:
            n_skipped += 1
            continue
            
        # 3. Reject likely scenes/interiors
        has_scene_keyword = any(kw in name for kw in scene_keywords) or \
                            any(kw in tag for tag in tags for kw in scene_keywords)
        if has_scene_keyword:
            n_skipped += 1
            continue

        # 4. Filter by vertex count and ensure it has categories
        if categories and vertex_count < max_vertex_count:
            sample_uids.append(uid)
        else:
            n_skipped += 1
            
    print(f"Selected {len(sample_uids)} UIDs meeting criteria. Skipped {n_skipped} UIDs.")
    if len(sample_uids) < num_objects:
        print(f"WARNING: Only found {len(sample_uids)} objects meeting criteria out of {len(subset_uids)} candidates.")
            
    return sample_uids

def download_objects_with_progress(uids, download_dir=GLB_DOWNLOAD_DIR, max_workers=16):
    """Custom download function to show a tqdm progress bar instead of printing new lines."""
    object_paths = objaverse._load_object_paths()
    out = {}
    to_download = []
    
    for uid in uids:
        if uid in object_paths:
            local_path = os.path.join(download_dir, object_paths[uid])
            if not os.path.exists(local_path):
                to_download.append((uid, object_paths[uid], local_path))
            else:
                out[uid] = local_path
                
    if to_download:
        def download_single(item):
            uid, obj_path, local_path = item
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{obj_path}"
            # Save to a temporary file first, then rename to avoid corrupted partial downloads
            tmp_local_path = local_path + ".tmp"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    urllib.request.urlretrieve(hf_url, tmp_local_path)
                    os.rename(tmp_local_path, local_path)
                    return uid, local_path
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        # Exponential backoff on 429 Too Many Requests
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(sleep_time)
                    else:
                        raise e
            raise Exception(f"Failed to download {uid} after {max_retries} retries due to rate limiting.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_single, item) for item in to_download]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading models"):
                uid, local_path = future.result()
                out[uid] = local_path
            
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_objects", type=int, default=10, help="Number of objects to download")
    parser.add_argument("--download", action="store_true", help="Download the selected objects")
    parser.add_argument("--uid_file", type=str, default="selected_uids.txt", help="File to save/load selected UIDs")
    args = parser.parse_args()

    uid_file = args.uid_file
    if os.path.exists(uid_file):
        print(f"Loading previously selected UIDs from {uid_file}...")
        with open(uid_file, "r") as f:
            selected_uids = [line.strip() for line in f if line.strip()]
        # Truncate if we want fewer objects than saved
        selected_uids = selected_uids[:args.num_objects]
    else:
        # Select objects
        selected_uids = select_uids(num_objects=args.num_objects)
        print(f"Saving selected UIDs to {uid_file}...")
        with open(uid_file, "w") as f:
            for uid in selected_uids:
                f.write(f"{uid}\n")
                
    print(f"Final selected count: {len(selected_uids)}")
    
    # Proceed to download:
    if args.download:
        print("Downloading selected objects...")
        objects = download_objects_with_progress(selected_uids)
