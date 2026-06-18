'''
Run with:
python training/train.py --config training/train_config.yml
python training/train.py --config training/train_config_71M.yml
'''

import os
import yaml
import argparse
import datetime
import torch
import torchvision
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lpips import LPIPS

from dataset import SceneDataset

from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------

def tone_map(img):  # as according to paper — clamp(log(I) / log(2), 0, 1)
    img = torch.log2(img.relu() + 1e-8)
    return torch.clamp(img, 0.0, 1.0)

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = torch.clamp(x, min=0.0, max=1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * torch.pow(x, 1/2.4) - a)

def tone_map_reinhard(x, exposure=1.0):
    x = x * exposure
    return x / (x + 1.0)

def to_uint8(x):
    """Convert float image [0, 1] to uint8 [0, 255]."""
    x = torch.clamp(x, 0.0, 1.0)
    return (x * 255).byte()

def hdr_to_ldr(x, exposure=1.0, to_uint8_output=True):
    """
    Convert HDR image to LDR (Reinhard tone-map → sRGB).

    Args:
        x:               Input HDR image tensor (any shape).
        exposure:        Pre-exposure multiplier before tone mapping.
        to_uint8_output: If True, output is uint8 [0, 255]; otherwise float [0, 1].
    """
    x = tone_map_reinhard(x, exposure=exposure)
    x = linear_to_srgb(x)
    if to_uint8_output:
        x = to_uint8(x)
    return x


# ---------------------------------------------------------------------------
# Visualization Utilities
# ---------------------------------------------------------------------------

def make_vis_grid(linear_rendered, gt_img, max_images=16, diff_amplify=5.0):
    """
    Build a side-by-side (pred | GT | diff) visualization grid.

    Args:
        linear_rendered: [bs, nv, H, W, C] float32 linear HDR tensor.
        gt_img:          [bs, nv, H, W, C] float32 linear HDR tensor.
        max_images:      Maximum number of (scene × view) panels to include.
        diff_amplify:    Scale factor applied to the abs-diff map so small
                         errors are clearly visible (default 5×).

    Returns:
        Uint8 grid tensor [C, H_total, W*3] suitable for writer.add_image.
    """
    bs, nv = linear_rendered.shape[:2]
    pred_flat = linear_rendered.detach().cpu().reshape(bs * nv, *linear_rendered.shape[2:])[:max_images]
    gt_flat   = gt_img.detach().cpu().reshape(bs * nv, *gt_img.shape[2:])[:max_images]

    # Tone-map to float [0, 1] first so the diff is in a perceptually uniform space
    pred_ldr = hdr_to_ldr(pred_flat.clamp(min=0.0), to_uint8_output=False)  # [N, H, W, C]
    gt_ldr   = hdr_to_ldr(gt_flat,                  to_uint8_output=False)

    # Absolute difference, amplified and clamped to [0, 1]
    diff = (pred_ldr - gt_ldr).abs().mul(diff_amplify).clamp(0.0, 1.0)

    # Convert all to uint8 and permute to [N, C, H, W]
    vis_pred = to_uint8(pred_ldr).permute(0, 3, 1, 2)
    vis_gt   = to_uint8(gt_ldr).permute(0, 3, 1, 2)
    vis_diff = to_uint8(diff).permute(0, 3, 1, 2)

    vis_img = torch.cat([vis_pred, vis_gt, vis_diff], dim=3)  # [N, C, H, W*3]
    grid = torchvision.utils.make_grid(vis_img.float(), nrow=1, normalize=False)
    return grid.byte()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Encapsulates the full training pipeline for RenderFormer."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.model_config_dict = config_dict.get('model', {})
        self.tc = config_dict.get('training', {})   # shorthand: training config

        # ---- Core objects ----
        self.model_config = RenderFormerConfig(**self.model_config_dict)
        self.device = torch.device(
            self.tc.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.resolution = self.tc.get('resolution', 512)

        self.model         = RenderFormer(self.model_config).to(self.device)
        self.ray_generator = RayGenerator().to(self.device)

        # ---- Training hyper-parameters ----
        self.epochs              = self.tc.get('epochs', 100)
        self.warmup_epochs       = self.tc.get('warmup_epochs', 10)
        self.log_viz_interval    = self.tc.get('log_viz_interval', 100)
        self.checkpoint_interval = self.tc.get('checkpoint_interval', 500)
        batch_size               = self.tc.get('batch_size', 1)
        val_batch_size           = self.tc.get('val_batch_size', batch_size)

        # ---- Datasets & dataloaders ----
        self.dataset_train = SceneDataset(
            data_dir=self.tc['data_dir'],
            resolution=self.resolution,
            split='train',
            max_dataset_size=self.tc.get('max_dataset_size', None),
        )
        self.dataset_val = SceneDataset(
            data_dir=self.tc['data_dir'],
            resolution=self.resolution,
            split='val',
            max_dataset_size=self.tc.get('max_dataset_size', None),
        )
        self.dataloader_train = DataLoader(
            self.dataset_train, batch_size=batch_size, shuffle=True
        )
        self.dataloader_val = DataLoader(
            self.dataset_val, batch_size=val_batch_size, shuffle=False
        )

        # ---- Fixed visualization batches (seeded for reproducibility) ----
        vis_size = min(16, batch_size)
        self.fixed_train_batch = self._get_seeded_batch(self.dataset_train, vis_size, seed=123)
        self.fixed_val_batch   = self._get_seeded_batch(self.dataset_val,   vis_size, seed=123)

        # ---- Optimizer & losses ----
        self.optimizer = AdamW(self.model.parameters(), lr=float(self.tc.get('lr', 1e-4)))
        self.loss_fn   = torch.nn.L1Loss()
        self.lpips_fn  = LPIPS(net='vgg').to(self.device)

        # ---- LR scheduler: linear warmup → cosine decay ----
        warmup_sched  = LinearLR(self.optimizer, start_factor=0.01, total_iters=self.warmup_epochs)
        cosine_sched  = CosineAnnealingLR(self.optimizer, T_max=self.epochs - self.warmup_epochs)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[self.warmup_epochs],
        )

        self.scaler = (
            torch.amp.GradScaler(self.device.type)
            if self.tc.get('use_amp', False) else None
        )

        # ---- Logging ----
        base_log_dir = self.tc.get(
            'log_dir',
            f'training/logs/{self.dataset_train.data_dir.split("/")[-1]}'
        )
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir        = os.path.join(base_log_dir, run_id)
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(self.log_dir, 'config.yaml'))
        print("Logging to:", self.log_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir)

    # ------------------------------------------------------------------
    # Data Helpers
    # ------------------------------------------------------------------

    def _get_seeded_batch(self, dataset, batch_size: int, seed: int = 42) -> dict:
        """
        Draw a single, reproducible batch from *dataset* using a seeded generator.
        The batch is moved to self.device immediately so it can be reused each epoch
        without repeated .to(device) calls.
        """
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,       # shuffle so the seeded draw gives variety
            num_workers=0,      # 0 workers keeps generator deterministic
            generator=g,
        )
        batch = next(iter(loader))
        return {k: v.to(self.device) for k, v in batch.items()}

    def _preprocess_batch(self, batch: dict) -> dict:
        """
        Move a raw dataset batch to device and compute all derived tensors
        (texture encoding, camera-coord transform, rays).
        """
        cfg = self.model_config
        dev = self.device

        triangles = batch['triangles'].to(dev)
        texture   = batch['texture'].to(dev)
        mask      = batch['mask'].to(dev)
        vn        = batch['vn'].to(dev)
        c2w       = batch['c2w'].to(dev)
        fov       = batch['fov'].to(dev)
        gt_img    = batch['gt_img'].to(dev)

        if fov.dim() == 2:
            fov = fov.unsqueeze(-1)

        bs, nv = c2w.shape[0], c2w.shape[1]

        if cfg.texture_encode_patch_size == 1 and texture.dim() == 5:
            texture = texture[:, :, :, 0, 0]
        if not cfg.use_ldr:
            texture_log = texture.clone()
            texture_log[:, :, -3:] = torch.log10(texture_log[:, :, -3:] + 1.)
        else:
            texture_log = texture

        if cfg.turn_to_cam_coord:
            c2w_reshaped       = c2w.reshape(-1, 4, 4)
            triangles_repeated = torch.repeat_interleave(triangles, nv, dim=0)
            tris_tf, c2w_tf, _ = trans_to_cam_coord(c2w_reshaped, triangles_repeated)
            c2w_tf  = c2w_tf.reshape(bs, nv, 4, 4)
            tris_tf = tris_tf.reshape(bs, nv, -1, 3, 3)
        else:
            tris_tf = triangles.unsqueeze(1).expand(-1, nv, -1, -1, -1)
            c2w_tf  = c2w

        rays_o, rays_d = self.ray_generator(c2w_tf, fov / 180. * torch.pi, self.resolution)

        return {
            'triangles':        triangles.reshape(bs, -1, 9),
            'texture_log':      texture_log,
            'mask':             mask,
            'vn':               vn.reshape(bs, -1, 9),
            'rays_o':           rays_o,
            'rays_d':           rays_d,
            'tri_vpos_view_tf': tris_tf.reshape(bs, nv, -1, 9),
            'gt_img':           gt_img,
        }

    # ------------------------------------------------------------------
    # Forward / Loss Helpers
    # ------------------------------------------------------------------

    def _forward(self, inputs: dict) -> torch.Tensor:
        """Run the model, optionally under AMP autocast."""
        kwargs = dict(
            rays_o=inputs['rays_o'],
            rays_d=inputs['rays_d'],
            tri_vpos_view_tf=inputs['tri_vpos_view_tf'],
            tf32_view_tf=False,
        )
        args = (inputs['triangles'], inputs['texture_log'], inputs['mask'], inputs['vn'])

        if self.scaler:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                return self.model(*args, **kwargs)
        return self.model(*args, **kwargs)

    def _compute_losses(self, rendered_imgs: torch.Tensor, gt_img: torch.Tensor):
        """
        Compute L1 (in log-HDR space) + LPIPS losses and PSNR.

        Args:
            rendered_imgs: Raw model output [bs, nv, C, H, W].
            gt_img:        Ground-truth HDR  [bs, nv, H, W, C].

        Returns:
            loss:            Combined scalar loss tensor.
            linear_rendered: Predicted images in linear HDR space [bs, nv, H, W, C].
            psnr:            Scalar float PSNR (no grad).
        """
        res = self.resolution
        raw = rendered_imgs.permute(0, 1, 3, 4, 2)  # [bs, nv, H, W, C]

        if not self.model_config.use_ldr:
            gt_log   = torch.log1p(gt_img.relu())
            loss_l1  = self.loss_fn(raw, gt_log)
            linear_rendered = torch.exp(torch.clamp(raw, max=10.0)) - 1.0
        else:
            loss_l1  = self.loss_fn(raw, gt_img)
            linear_rendered = raw

        # LPIPS in tone-mapped space
        tm_pred = tone_map(linear_rendered)
        tm_gt   = tone_map(gt_img)
        tm_pred_nchw = tm_pred.view(-1, res, res, 3).permute(0, 3, 1, 2) * 2.0 - 1.0
        tm_gt_nchw   = tm_gt.view(-1,   res, res, 3).permute(0, 3, 1, 2) * 2.0 - 1.0

        loss_lpips = self.lpips_fn(tm_pred_nchw, tm_gt_nchw).mean()
        loss = loss_l1 + 0.05 * loss_lpips

        with torch.no_grad():
            mse  = torch.nn.functional.mse_loss(linear_rendered, gt_img)
            psnr = -10.0 * torch.log10(mse + 1e-8)

        return loss, linear_rendered, psnr.item()

    # ------------------------------------------------------------------
    # Logging Helpers
    # ------------------------------------------------------------------

    def _log_scalars(self, prefix: str, metrics: dict, epoch: int):
        """Write a dict of {tag: value} scalars to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, epoch)

    def _log_vis_grid(self, tag: str, inputs: dict, epoch: int):
        """
        Run the model on *inputs* (already preprocessed), build a grid, and log it.
        Must be called inside a torch.no_grad() context.
        """
        rendered = self._forward(inputs)
        _, linear, _ = self._compute_losses(rendered, inputs['gt_img'])
        grid = make_vis_grid(linear, inputs['gt_img'], max_images=16)
        self.writer.add_image(tag, grid, epoch)

    # ------------------------------------------------------------------
    # Train / Val Steps
    # ------------------------------------------------------------------

    def _train_epoch(self) -> tuple[float, float]:
        """Run one full training epoch. Returns (avg_loss, avg_psnr)."""
        self.model.train()
        total_loss, total_psnr = 0.0, 0.0

        for batch in self.dataloader_train:
            inputs = self._preprocess_batch(batch)
            rendered = self._forward(inputs)
            loss, _, psnr = self._compute_losses(rendered, inputs['gt_img'])

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_psnr += psnr

        n = len(self.dataloader_train)
        return total_loss / n, total_psnr / n

    def _validate(self, epoch: int):
        """
        Evaluate on the full validation set and log metrics + visualizations.
        Switches the model to eval mode and back before returning.
        """
        self.model.eval()
        total_loss, total_psnr = 0.0, 0.0

        with torch.no_grad():
            # --- Full-set metrics ---
            for batch in self.dataloader_val:
                inputs = self._preprocess_batch(batch)
                rendered = self._forward(inputs)
                loss, _, psnr = self._compute_losses(rendered, inputs['gt_img'])
                total_loss += loss.item()
                total_psnr += psnr

            n = len(self.dataloader_val)
            avg_val_loss = total_loss / n
            avg_val_psnr = total_psnr / n

            self._log_scalars('Val', {'Loss': avg_val_loss, 'PSNR': avg_val_psnr}, epoch)
            print(f"  Val  Loss: {avg_val_loss:.4f}  PSNR: {avg_val_psnr:.4f}")

            # --- Visualizations on fixed seeded batches ---
            train_inputs = self._preprocess_batch(self.fixed_train_batch)
            self._log_vis_grid('Train/Rendered_vs_GT', train_inputs, epoch)

            val_inputs = self._preprocess_batch(self.fixed_val_batch)
            self._log_vis_grid('Val/Rendered_vs_GT', val_inputs, epoch)

        self.model.train()

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full training loop."""
        for epoch in tqdm(range(self.epochs)):
            avg_train_loss, avg_train_psnr = self._train_epoch()

            self.scheduler.step()
            self._log_scalars('Train', {
                'LR':   self.optimizer.param_groups[0]['lr'],
                'Loss': avg_train_loss,
                'PSNR': avg_train_psnr,
            }, epoch)

            # Validation + visualization
            if (epoch + 1) % self.log_viz_interval == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}  PSNR: {avg_train_psnr:.4f}")
                self._validate(epoch)

            # Checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch':                epoch + 1,
                    'model_state_dict':     self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss':                 avg_train_loss,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        self.writer.close()
        print("Training complete.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/train_config.yml")
    args = parser.parse_args()

    Trainer(args.config).run()