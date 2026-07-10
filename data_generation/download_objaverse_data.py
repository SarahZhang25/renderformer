import objaverse
import os
import random
import trimesh

random.seed(42)
MAX_VERTEX_COUNT = 50000

# Set base paths to your custom directory
objaverse.BASE_PATH = "/home/sazhang/.objaverse"
objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

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

if __name__ == "__main__":
    # Select up to 10000 objects
    selected_uids = select_uids(num_objects=10)
    print(f"Final selected count: {len(selected_uids)}")
    
    # Proceed to download:
    # objects = objaverse.load_objects(uids=selected_uids)
