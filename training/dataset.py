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
        split_proportion: float = 0.9
    ):
        self.data_dir = data_dir
        self.resolution = resolution

        # Find all completed cases (must have .h5)
        self.files = glob.glob(os.path.join(data_dir, "*.h5"))
        self.files.sort()
        if max_dataset_size is not None and max_dataset_size < len(self.files):
            self.files = self.files[:max_dataset_size]

        # Shuffle with a fixed seed
        if split == "all":
            print(f"[{split}] Using all {len(self.files)} samples in {data_dir}")
        else:
            assert split in ['train', 'val'], "split must be 'train', 'val', or 'all'"
            rng = np.random.RandomState(42)
            rng.shuffle(self.files)
            
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
