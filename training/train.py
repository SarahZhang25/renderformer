'''
Run with:
python training/train.py --config training/train_config.yml
'''

import os
import yaml
import argparse
import datetime
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lpips import LPIPS

from dataset import SingleSceneDataset

from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord

def tone_map(img):
    # Tone map to clamp(log(I) / log(2), 0, 1)
    img = torch.log2(img.relu() + 1e-8)
    return torch.clamp(img, 0.0, 1.0)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/train_config.yml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    model_config_dict = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    
    config = RenderFormerConfig(**model_config_dict)
    device = torch.device(training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    model = RenderFormer(config).to(device)
    model.train()
    
    ray_generator = RayGenerator().to(device)
    
    dataset = SingleSceneDataset(training_config['h5_path'], training_config['gt_dir'], resolution=training_config.get('resolution', 512))
    
    dataloader = DataLoader(dataset, batch_size=training_config.get('batch_size', 1), shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=float(training_config.get('lr', 1e-4)))
    loss_fn = torch.nn.L1Loss()
    lpips_fn = LPIPS(net='vgg').to(device)
    
    # LR scheduling: Warmup + Cosine decay
    # Warm up linearly over the first warmup_epochs.
    # Automatically kick in the Cosine decay for the remaining epochs
    epochs = training_config.get('epochs', 100)
    warmup_epochs = training_config.get('warmup_epochs', 10)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)  # decay
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    resolution = training_config.get('resolution', 512)
    
    scaler = torch.amp.GradScaler(device.type) if training_config.get('use_amp', False) else None

    # Logging
    base_log_dir = training_config.get('log_dir', 'runs/renderformer_exp')
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_log_dir, run_id)
    print("Logging to:", log_dir)

    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(log_dir, 'config.yaml'))


    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        for batch in dataloader:
            triangles = batch['triangles'].to(device)
            texture = batch['texture'].to(device)
            mask = batch['mask'].to(device)
            vn = batch['vn'].to(device)
            c2w = batch['c2w'].unsqueeze(1).to(device) # Add view dim: [bs, 1, 4, 4]
            fov = batch['fov'].unsqueeze(-1).to(device) if batch['fov'].dim() == 2 else batch['fov'].to(device)
            if fov.dim() == 2:
                fov = fov.unsqueeze(-1)
            gt_img = batch['gt_img'].to(device)
            
            bs, nv = c2w.shape[0], c2w.shape[1]
            
            # Preprocess inputs
            if config.texture_encode_patch_size == 1 and texture.dim() == 5:
                texture = texture[:, :, :, 0, 0]
            if not config.use_ldr:
                texture_log = texture.clone()
                texture_log[:, :, -3:] = torch.log10(texture_log[:, :, -3:] + 1.)
            else:
                texture_log = texture
                
            if config.turn_to_cam_coord:
                c2w_reshaped = c2w.reshape(-1, 4, 4)
                triangles_repeated = torch.repeat_interleave(triangles, nv, dim=0)
                tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(c2w_reshaped, triangles_repeated)
                c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
                tris_for_view_tf = tris_for_view_tf.reshape(bs, nv, -1, 3, 3)
            else:
                tris_for_view_tf = triangles.unsqueeze(1).expand(-1, nv, -1, -1, -1)
                c2w_for_view_tf = c2w

            rays_o, rays_d = ray_generator(c2w_for_view_tf, fov / 180. * torch.pi, resolution)
            
            tf32_view_tf = False
            
            # Forward pass
            if scaler:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    rendered_imgs = model(
                        triangles.reshape(bs, -1, 9),
                        texture_log,
                        mask,
                        vn.reshape(bs, -1, 9),
                        rays_o=rays_o,
                        rays_d=rays_d,
                        tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                        tf32_view_tf=tf32_view_tf,
                    )
            else:
                rendered_imgs = model(
                    triangles.reshape(bs, -1, 9),
                    texture_log,
                    mask,
                    vn.reshape(bs, -1, 9),
                    rays_o=rays_o,
                    rays_d=rays_d,
                    tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                    tf32_view_tf=tf32_view_tf,
                )
            

            # Process output
            # rendered_imgs: [bs, nv, C, H, W] -> [bs, nv, H, W, C]
            raw_rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)
            
            gt_hdr = gt_img.unsqueeze(1)
            
            if not config.use_ldr:
                # The model natively predicts log10(HDR + 1), so supervise it directly in log HDR space
                gt_log = torch.log10(gt_hdr.relu() + 1.0)
                loss_l1 = loss_fn(raw_rendered_imgs, gt_log)
                
                rendered_imgs = torch.pow(10., raw_rendered_imgs) - 1.
            else:
                loss_l1 = loss_fn(raw_rendered_imgs, gt_hdr)
                rendered_imgs = raw_rendered_imgs
            
            # Tone map for LPIPS calculation according to paper
            tm_rendered = tone_map(rendered_imgs)
            tm_gt = tone_map(gt_hdr)
            
            # Convert to [N, C, H, W] and [-1, 1] range for LPIPS
            tm_rendered_nchw = tm_rendered.view(-1, resolution, resolution, 3).permute(0, 3, 1, 2) * 2.0 - 1.0
            tm_gt_nchw = tm_gt.view(-1, resolution, resolution, 3).permute(0, 3, 1, 2) * 2.0 - 1.0
            
            loss_lpips = lpips_fn(tm_rendered_nchw, tm_gt_nchw).mean()
            loss = loss_l1 + 0.05 * loss_lpips
            
            with torch.no_grad():
                mse = torch.nn.functional.mse_loss(rendered_imgs, gt_img.unsqueeze(1))
                psnr = -10.0 * torch.log10(mse + 1e-8)
                epoch_psnr += psnr.item()
            
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/PSNR', avg_psnr, epoch)
        
        # Log images for visualization
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - PSNR: {avg_psnr:.4f}")
            with torch.no_grad():
                img_hdr = rendered_imgs[0, 0].detach().cpu()
                img_ldr = torch.clamp(img_hdr, 0.0, 1.0)
                img_ldr = (img_ldr * 255).to(torch.uint8)
                img_ldr = img_ldr.permute(2, 0, 1)

                gt_hdr = gt_img[0].detach().cpu()
                gt_ldr = torch.clamp(gt_hdr, 0.0, 1.0)
                gt_ldr = (gt_ldr * 255).to(torch.uint8)
                gt_ldr = gt_ldr.permute(2, 0, 1)

                combined_ldr = torch.cat([img_ldr, gt_ldr], dim=2)
                writer.add_image('Train/Rendered_vs_GT', combined_ldr, epoch)

if __name__ == '__main__':
    train()