import h5py
import imageio
import numpy as np
import os
import torch

from torch.utils.data import Dataset

class SingleSceneDataset(Dataset):
    def __init__(self, h5_path, gt_dir, resolution=512):
        self.h5_path = h5_path
        self.gt_dir = gt_dir
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
        return self.num_views

    def __getitem__(self, idx):
        c2w = self.c2w[idx] # [4, 4]
        fov = self.fov[idx:idx+1]
        
        gt_path = os.path.join(self.gt_dir, f"{self.base_name}_view_{idx}.exr")
        gt_img = imageio.v3.imread(gt_path).astype(np.float32)
        gt_img = torch.from_numpy(gt_img)
        
        return {
            'triangles': self.triangles,
            'texture': self.texture,
            'mask': self.mask,
            'vn': self.vn,
            'c2w': c2w,
            'fov': fov,
            'gt_img': gt_img
        }
