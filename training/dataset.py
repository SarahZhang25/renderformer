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
        max_train_scenes: int = None,
        split: str = "all",
        split_ratio: float = 0.9,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        augment_rotation: bool = False,
        augment_permute: bool = False
    ):
        if isinstance(data_dir, str):
            self.data_dirs = [data_dir]
        else:
            self.data_dirs = data_dir
            
        self.image_res = image_res
        self.num_views_per_scene = 4

        
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
        total_samples = total_scenes * self.num_views_per_scene
        
        if max_dataset_size is not None:
            max_scenes = max_dataset_size // self.num_views_per_scene
            if max_scenes < total_scenes:
                total_scenes = max_scenes
                total_samples = total_scenes * self.num_views_per_scene

        sample_order = np.arange(total_samples, dtype=np.int32)
        
        if split == "all":
            print(f"[{split}] Using all {len(sample_order)} samples across {len(self.chunk_files)} chunks")
        else:
            assert split in ['train', 'val'], "split must be 'train', 'val', or 'all'"
            split_idx = int(total_scenes * split_ratio) * self.num_views_per_scene
            if split == 'train':
                sample_order = sample_order[:split_idx]
                # max_train_scenes shrinks the TRAINING set only. max_dataset_size cannot
                # be used for this: it truncates before the split, so lowering it shrinks
                # the validation set too and a dataset-scaling curve would compare points
                # measured on different held-out scenes. Truncating here keeps validation
                # fixed, and taking a prefix makes the smaller training sets nested
                # subsets of the larger ones, so consecutive points on the curve differ
                # only in how many scenes were added.
                if max_train_scenes is not None:
                    keep = max_train_scenes * self.num_views_per_scene
                    if keep < len(sample_order):
                        sample_order = sample_order[:keep]
            else:
                sample_order = sample_order[split_idx:]
                
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        # Same exact augmentation as the NMR path, so a run with augmentation on stays a
        # like-for-like comparison between the two architectures.
        self.split = split
        self.augment_rotation = bool(augment_rotation) and split == 'train'
        self.augment_permute = bool(augment_permute) and split == 'train'
        self._epoch = 0
        self.original_sample_order = sample_order.copy()
        self.sample_order = sample_order
        
        # Initial shuffle
        self.set_epoch(0)
        
        print(f"[{split}] Found {len(self.sample_order)} samples across {len(self.chunk_files)} chunks from {len(self.data_dirs)} directories")

        # Lazily store opened H5 handles per worker to avoid multiprocess fork issues
        self._h5_handles = {}

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        if not self.shuffle:
            return
            
        rng = np.random.RandomState(self.shuffle_seed + epoch)
        
        # Determine exact chunk boundaries within our current split's sample_order
        global_chunk_boundaries = self.chunk_offsets * self.num_views_per_scene
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
        """Convert a position in sample_order to (chunk_path, scene_name, view_idx)."""
        global_sample = int(self.sample_order[idx])
        scene_idx = global_sample // self.num_views_per_scene
        view_idx = global_sample % self.num_views_per_scene
        # Binary search to find which chunk this scene belongs to
        chunk_idx = int(np.searchsorted(self.chunk_offsets, scene_idx, side='right')) - 1
        local_scene_idx = scene_idx - int(self.chunk_offsets[chunk_idx])
        chunk_path, scene_names = self.chunk_meta[chunk_idx]
        return chunk_path, scene_names[local_scene_idx], view_idx

    def __len__(self):
        return len(self.sample_order)

    def __getitem__(self, idx):
        chunk_file, scene_name, view_idx = self._decode_idx(idx)
        f = self._get_h5_file(chunk_file)
        grp = f[scene_name]
        
        triangles = torch.from_numpy(np.array(grp['triangles'])).float()

        # Slice inside HDF5 rather than after: this used to read the whole
        # (2184, 13, 32, 32) float16 texture -- ~58 MB per sample -- and then keep only
        # [:, :, 0, 0] (~57 KB). With chunks=(273, 2, 8, 8) the sliced read touches
        # 56 of 896 chunks instead of all of them.
        tex_ds = grp['texture']
        if tex_ds.ndim == 4:
            texture = torch.from_numpy(tex_ds[:, :, 0, 0]).float()
        else:
            texture = torch.from_numpy(tex_ds[:]).float()

        vn = torch.from_numpy(np.array(grp['vn'])).float()
        c2w_np = np.array(grp['c2w'])
        fov_np = np.array(grp['camera_fov'])

        # Same idea for the target image: hdr_target_image is (V, 512, 512, 4) float32
        # = 16 MB, and only one view is ever used. Read that view only.
        img_ds = grp['hdr_target_image']
        if img_ds.ndim == 4:
            V = img_ds.shape[0]
            v_idx = min(view_idx, V - 1)
            gt_img_np = img_ds[v_idx]
            c2w_np = c2w_np[v_idx]
            if isinstance(fov_np, np.ndarray) and fov_np.shape[0] == V:
                fov_np = fov_np[v_idx]
        else:
            gt_img_np = img_ds[:]

        c2w = torch.from_numpy(c2w_np).float()
        fov = torch.tensor(fov_np).float()
        mask = torch.ones(triangles.shape[0], dtype=torch.bool)
        
        # Load embedded HDR image directly from HDF5
        gt_img = torch.from_numpy(gt_img_np).float()
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[..., :3]
        
        # Resize if necessary
        if gt_img.shape[0] != self.image_res:
            # Permute to shape expected by interpolate: [C, H, W]
            gt_img = gt_img.permute(2, 0, 1).unsqueeze(0) 
            gt_img = torch.nn.functional.interpolate(gt_img, size=(self.image_res, self.image_res), mode='bilinear', align_corners=False)
            # Permute back to shape: [H, W, C]
            gt_img = gt_img.squeeze(0).permute(1, 2, 0)

        sample = {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'vn': vn,
            'c2w': c2w,
            'camera_fov': fov,
            'gt_img': gt_img
        }

        if self.augment_rotation or self.augment_permute:
            from training.augment import (permute_entities, random_rotation,
                                          rotate_sample, sample_rng)
            rng = sample_rng(self.shuffle_seed, self._epoch, idx)
            if self.augment_rotation:
                rotate_sample(sample, random_rotation(rng), ('triangles', 'vn'))
            if self.augment_permute:
                permute_entities(sample, ('triangles', 'texture', 'mask', 'vn'), rng)

        return sample

    def __del__(self):
        # Close all H5 handles on destruction
        for f in self._h5_handles.values():
            f.close()
