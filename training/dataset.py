import glob
import h5py
import imageio
import numpy as np
import os
import torch

from torch.utils.data import Dataset

def scene_collate_fn(batch):
    max_tris = max(item['triangles'].shape[0] for item in batch)
    
    batched_data = {
        'triangles': [],
        'texture': [],
        'mask': [],
        'vn': [],
        'c2w': [],
        'fov': [],
        'gt_img': []
    }
    
    for item in batch:
        num_tris = item['triangles'].shape[0]
        pad_size = max_tris - num_tris
        
        if pad_size > 0:
            triangles = torch.cat([item['triangles'], item['triangles'].new_zeros(pad_size, *item['triangles'].shape[1:])], dim=0)
            texture = torch.cat([item['texture'], item['texture'].new_zeros(pad_size, *item['texture'].shape[1:])], dim=0)
            vn = torch.cat([item['vn'], item['vn'].new_zeros(pad_size, *item['vn'].shape[1:])], dim=0)
            mask = torch.cat([item['mask'], item['mask'].new_zeros(pad_size, *item['mask'].shape[1:])], dim=0)
        else:
            triangles = item['triangles']
            texture = item['texture']
            vn = item['vn']
            mask = item['mask']
            
        batched_data['triangles'].append(triangles)
        batched_data['texture'].append(texture)
        batched_data['mask'].append(mask)
        batched_data['vn'].append(vn)
        batched_data['c2w'].append(item['c2w'])
        batched_data['fov'].append(item['camera_fov'])
        batched_data['gt_img'].append(item['gt_img'])
        
    return {k: torch.stack(v, dim=0) for k, v in batched_data.items()}

class H5SceneDataset(Dataset):
    """
    Dataloader for the chunked RF-format HDF5 datasets.
    Supports single-writer-multiple-reader (SWMR) for efficient multiprocess loading.
    Embeds the target HDR images directly inside the H5, avoiding separate disk reads.
    """
    def __init__(
        self,
        data_dir, # can be a string or a list of strings
        image_res: int = 128,
        max_dataset_size: int = None,
        split: str = "all",
        split_ratio: float = 0.9,
        shuffle: bool = True,
        shuffle_seed: int = 42
    ):
        if isinstance(data_dir, str):
            self.data_dirs = [data_dir]
        else:
            self.data_dirs = data_dir
            
        self.image_res = image_res
        
        # Glob all rf-formatted chunk files
        self.chunk_files = []
        for d in self.data_dirs:
            if d.endswith('.h5'):
                self.chunk_files.append(d)
            else:
                self.chunk_files.extend(glob.glob(os.path.join(d, "rf_dataset_chunk_*.h5")))
        self.chunk_files = sorted(list(set(self.chunk_files)))
        
        # Build compact per-chunk metadata: one entry per chunk, not per sample.
        self.chunk_meta = []  # list of (chunk_path, scene_names_list)
        for chunk_file in self.chunk_files:
            with h5py.File(chunk_file, 'r') as f:
                scene_names = list(f.keys())
            self.chunk_meta.append((chunk_file, scene_names))
            
        # chunk_offsets[i] = first global scene index in chunk i (in scenes)
        scene_counts = np.array([len(names) for _, names in self.chunk_meta], dtype=np.int64)
        self.chunk_offsets = np.concatenate([[0], np.cumsum(scene_counts)]).astype(np.int64)
        total_scenes = int(self.chunk_offsets[-1])
        
        if max_dataset_size is not None and max_dataset_size < total_scenes:
            total_scenes = max_dataset_size

        sample_order = np.arange(total_scenes, dtype=np.int32)
        
        if split == "all":
            print(f"[{split}] Using all {len(sample_order)} scenes across {len(self.chunk_files)} chunks")
        else:
            assert split in ['train', 'val'], "split must be 'train', 'val', or 'all'"
            split_idx = int(total_scenes * split_ratio)
            if split == 'train':
                sample_order = sample_order[:split_idx]
            else:
                sample_order = sample_order[split_idx:]
                
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.original_sample_order = sample_order.copy()
        self.sample_order = sample_order
        
        # Initial shuffle
        self.set_epoch(0)
        
        print(f"[{split}] Found {len(self.sample_order)} scenes across {len(self.chunk_files)} chunks from {len(self.data_dirs)} directories")

        # Lazily store opened H5 handles per worker to avoid multiprocess fork issues
        self._h5_handles = {}

    def set_epoch(self, epoch: int):
        if not self.shuffle:
            return
            
        rng = np.random.RandomState(self.shuffle_seed + epoch)
        
        # Determine exact chunk boundaries within our current split's sample_order
        global_chunk_boundaries = self.chunk_offsets
        # Find where these global boundaries land inside our sliced sample_order
        local_boundaries = np.searchsorted(self.original_sample_order, global_chunk_boundaries)
        # Ensure 0 and len(sample_order) are included, and remove duplicates
        local_boundaries = np.unique(np.clip(local_boundaries, 0, len(self.original_sample_order)))
        
        blocks = []
        for i in range(len(local_boundaries) - 1):
            start = local_boundaries[i]
            end = local_boundaries[i+1]
            if start < end:
                block = self.original_sample_order[start:end].copy()
                rng.shuffle(block)
                blocks.append(block)
                
        rng.shuffle(blocks)
        self.sample_order = np.concatenate(blocks)

    def _get_h5_file(self, chunk_path):
        if chunk_path not in self._h5_handles:
            # swmr=True enables Single Writer Multiple Reader, safe for multiprocess dataloading
            self._h5_handles[chunk_path] = h5py.File(chunk_path, 'r', swmr=True)
        return self._h5_handles[chunk_path]

    def _decode_idx(self, idx):
        """Convert a position in sample_order to (chunk_path, scene_name)."""
        scene_idx = int(self.sample_order[idx])
        # Binary search to find which chunk this scene belongs to
        chunk_idx = int(np.searchsorted(self.chunk_offsets, scene_idx, side='right')) - 1
        local_scene_idx = scene_idx - int(self.chunk_offsets[chunk_idx])
        chunk_path, scene_names = self.chunk_meta[chunk_idx]
        return chunk_path, scene_names[local_scene_idx]

    def __len__(self):
        return len(self.sample_order)

    def __getitem__(self, idx):
        chunk_file, scene_name = self._decode_idx(idx)
        f = self._get_h5_file(chunk_file)
        grp = f[scene_name]
        
        triangles = torch.from_numpy(np.array(grp['triangles'])).float()
        texture = torch.from_numpy(np.array(grp['texture'])).float()
        if texture.dim() == 4:
            texture = texture[:, :, 0, 0]
        vn = torch.from_numpy(np.array(grp['vn'])).float()
        c2w = torch.from_numpy(np.array(grp['c2w'])).float()
        fov = torch.from_numpy(np.array(grp['camera_fov'])).float()
        mask = torch.ones(triangles.shape[0], dtype=torch.bool)
        
        # Load embedded HDR image directly from HDF5
        gt_img = torch.from_numpy(grp['hdr_target_image'][:]).float()
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[..., :3]
        
        # Resize if necessary
        if gt_img.shape[1] != self.image_res:
            # Permute to shape expected by interpolate: [N, C, H, W] (where N = num_views)
            gt_img = gt_img.permute(0, 3, 1, 2) 
            gt_img = torch.nn.functional.interpolate(gt_img, size=(self.image_res, self.image_res), mode='bilinear', align_corners=False)
            # Permute back to shape: [num_views, H, W, C]
            gt_img = gt_img.permute(0, 2, 3, 1)

        return {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'vn': vn,
            'c2w': c2w,
            'camera_fov': fov,
            'gt_img': gt_img
        }

    def __del__(self):
        # Close all H5 handles on destruction
        for f in self._h5_handles.values():
            f.close()
