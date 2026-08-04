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
        batched_data['fov'].append(item['fov'])
        batched_data['gt_img'].append(item['gt_img'])
        
    return {k: torch.stack(v, dim=0) for k, v in batched_data.items()}

class SceneDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        resolution: int = 128,
        max_dataset_size = None,
        split: str = "all",
        split_proportion: float = 0.9,
        shuffle: bool = True,
        shuffle_seed: int = 42
    ):
        self.data_dir = data_dir
        self.resolution = resolution

        # Find all completed cases (must have .h5)
        # Assumes each scene gets its own h5
        self.files = glob.glob(os.path.join(data_dir, "scene*.h5"))
        self.files.sort()

        if shuffle:
            rng = np.random.RandomState(shuffle_seed)
            rng.shuffle(self.files)

        if max_dataset_size is not None and max_dataset_size < len(self.files):
            self.files = self.files[:max_dataset_size]

        # Shuffle with a fixed seed
        if split == "all":
            print(f"[{split}] Using all {len(self.files)} samples in {data_dir}")
        else:
            assert split in ['train', 'val'], "split must be 'train', 'val', or 'all'"
            split_idx = int(len(self.files) * split_proportion)
            if split == 'train':
                self.files = self.files[:split_idx]
            else:
                self.files = self.files[split_idx:]
                
                
        print(f"[{split}] Found {len(self.files)} samples in {data_dir}")


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as f:
            triangles = torch.from_numpy(np.array(f['triangles'])).float()
            texture = torch.from_numpy(np.array(f['texture'])).float()
            if texture.dim() == 4:
                texture = texture[:, :, 0, 0]
            vn = torch.from_numpy(np.array(f['vn'])).float()
            c2w = torch.from_numpy(np.array(f['c2w'])).float()
            fov = torch.from_numpy(np.array(f['fov'])).float()
            mask = torch.ones(triangles.shape[0], dtype=torch.bool)
        
        num_views = c2w.shape[0]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        gt_imgs = []
        for i in range(num_views):
            gt_path = os.path.join(self.data_dir, f"{base_name}_{i}.exr")
            gt_img = imageio.v3.imread(gt_path).astype(np.float32)
            gt_img = torch.from_numpy(gt_img)
            # Keep only RGB channels if there is an alpha channel
            if gt_img.shape[-1] == 4:
                gt_img = gt_img[..., :3]
            gt_imgs.append(gt_img)

        # Stack all views to shape: [num_views, H, W, C]
        gt_img = torch.stack(gt_imgs, dim=0)
        
        # Resize if necessary
        if gt_img.shape[1] != self.resolution:
            # Permute to shape expected by interpolate: [N, C, H, W] (where N = num_views)
            gt_img = gt_img.permute(0, 3, 1, 2) 
            gt_img = torch.nn.functional.interpolate(gt_img, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
            # Permute back to shape: [num_views, H, W, C]
            gt_img = gt_img.permute(0, 2, 3, 1)

        return {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'vn': vn,
            'c2w': c2w,
            'fov': fov,
            'gt_img': gt_img
        }


class SingleSceneDataset(Dataset):
    def __init__(self, data_dir, resolution=512):
        self.data_dir = data_dir
        self.h5_path = glob.glob(os.path.join(data_dir, "*.h5"))[0]
        
        self.resolution = resolution

        with h5py.File(self.h5_path, 'r') as f:
            self.triangles = torch.from_numpy(np.array(f['triangles'])).float()
            self.num_tris = self.triangles.shape[0]
            self.texture = torch.from_numpy(np.array(f['texture'])).float()
            if self.texture.dim() == 4:
                self.texture = self.texture[:, :, 0, 0]
            self.vn = torch.from_numpy(np.array(f['vn'])).float()
            self.c2w = torch.from_numpy(np.array(f['c2w'])).float()
            self.fov = torch.from_numpy(np.array(f['fov'])).float()
            self.mask = torch.ones(self.num_tris, dtype=torch.bool)
        
        self.num_views = self.c2w.shape[0]
        self.base_name = os.path.splitext(os.path.basename(self.h5_path))[0]
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        gt_imgs = []
        for i in range(self.num_views):
            gt_path = os.path.join(self.data_dir, f"{self.base_name}_view_{i}.exr")
            gt_img = imageio.v3.imread(gt_path).astype(np.float32)
            gt_img = torch.from_numpy(gt_img)
            # Keep only RGB channels if there is an alpha channel
            if gt_img.shape[-1] == 4:
                gt_img = gt_img[..., :3]
            gt_imgs.append(gt_img)
            
        # Stack all views to shape: [num_views, H, W, C]
        gt_img = torch.stack(gt_imgs, dim=0)
        
        # Resize if necessary
        if gt_img.shape[1] != self.resolution:
            # Permute to shape expected by interpolate: [N, C, H, W] (where N = num_views)
            gt_img = gt_img.permute(0, 3, 1, 2)
            gt_img = torch.nn.functional.interpolate(gt_img, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
            # Permute back to shape: [num_views, H, W, C]
            gt_img = gt_img.permute(0, 2, 3, 1)
        
        return {
            'triangles': self.triangles,
            'texture': self.texture,
            'mask': self.mask,
            'vn': self.vn,
            'c2w': self.c2w,
            'fov': self.fov,
            'gt_img': gt_img
        }

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
        fov = torch.from_numpy(np.array(grp['fov'])).float()
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
            'fov': fov,
            'gt_img': gt_img
        }

    def __del__(self):
        # Close all H5 handles on destruction
        for f in self._h5_handles.values():
            f.close()
