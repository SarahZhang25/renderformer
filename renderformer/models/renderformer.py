import torch
from torch import nn
from torch.amp import autocast

from huggingface_hub import PyTorchModelHubMixin

from renderformer.renderformer.encodings.nerf_encoding import NeRFEncoding
from renderformer.renderformer.layers.attention import TransformerEncoder
from renderformer.renderformer.models.view_transformer import ViewTransformer
from renderformer.renderformer.models.config import RenderFormerConfig


class RenderFormer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: RenderFormerConfig):
        super(RenderFormer, self).__init__()
        self.config = config

        if self.config.pe_type == 'nerf':
            # vertex PE and projections
            self.tri_vpos_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=self.config.vertex_pe_num_freqs,
                include_input=True
            )
            # triangle pe projection
            self.tri_encoding_proj = nn.Linear(
                self.tri_vpos_pe.get_out_dim(),
                self.config.latent_dim
            )
            # reuse this config for ablation...
            if self.config.vn_encoder_norm_type == 'layer_norm':
                self.tri_encoding_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'rms_norm':
                self.tri_encoding_norm = nn.RMSNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'none':
                self.tri_encoding_norm = nn.Identity()
            self.rope_dim = None
        elif self.config.pe_type == 'rope':
            self.rope_dim = self.config.vertex_pe_num_freqs
            # optional alternative where we clamp rope dim
            # head_dim = self.config.latent_dim // self.config.num_heads
            # max_rope_dim = ((head_dim // 2) // 9) * 2
            # self.rope_dim = min(self.config.vertex_pe_num_freqs, max_rope_dim)
        else:
            raise ValueError(f"Invalid positional encoding type: {self.config.pe_type}")

        if self.config.use_vn_encoder:
            self.vn_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=self.config.vn_pe_num_freqs,
                include_input=True
            )
            self.vn_encoding_proj = nn.Linear(
                self.vn_pe.get_out_dim(),
                self.config.latent_dim
            )
            if self.config.vn_encoder_norm_type == 'layer_norm':
                self.vn_encoder_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'rms_norm':
                self.vn_encoder_norm = nn.RMSNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'none':
                self.vn_encoder_norm = nn.Identity()
            else:
                raise ValueError(f"Invalid vertex normal encoder normalization type: {self.config.vn_encoder_norm_type}")

        # texture encoder
        self.texture_encoder = nn.Linear(
            self.config.texture_channels * self.config.texture_encode_patch_size * self.config.texture_encode_patch_size,
            self.config.latent_dim
        )
        if self.config.texture_encoder_norm_type == 'layer_norm':
            self.texture_encoder_norm = nn.LayerNorm(self.config.latent_dim)
        elif self.config.texture_encoder_norm_type == 'rms_norm':
            self.texture_encoder_norm = nn.RMSNorm(self.config.latent_dim)
        else:
            raise ValueError(f"Invalid texture encoder normalization type: {self.config.texture_encoder_norm_type}")

        # learnable tokens
        self.tri_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim))
        self.reg_tokens = nn.Parameter(torch.randn(1, self.config.num_register_tokens, self.config.latent_dim))
        self.skip_token_num = self.config.num_register_tokens

        # core radiosity transformer
        self.transformer = TransformerEncoder(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.latent_dim,
            ffn_hidden_dim=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            norm_first=self.config.norm_first,
            rope_dim=self.rope_dim,
            rope_type=self.config.rope_type,
            bias=self.config.bias,
            qk_norm=self.config.view_indep_qk_norm,
            rope_double_max_freq=self.config.rope_double_max_freq
        )

        # view transformer
        self.view_transformer = ViewTransformer(config)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    @autocast(device_type="cuda", dtype=torch.float32)  # avoid bf16 for its low precision
    def process_tri_vpos_list(self, tri_vpos_list, valid_mask):
        """
        Process tri_vpos_list for RoPE positional encoding.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param valid_mask: [batch_size, max_num_tri]
        :return: processed tri_vpos_list, updated valid_mask
        """
        mask_weight = (valid_mask.float() / (valid_mask.sum(dim=1, keepdim=True) + 1e-5))[..., None]
        weighted_tri_pos = mask_weight * tri_vpos_list
        center_pos = weighted_tri_pos.sum(dim=1).reshape(-1, 3, 3).mean(dim=1, keepdim=True).repeat(1, self.skip_token_num, 3)
        tri_vpos_list = torch.cat([center_pos, tri_vpos_list], dim=1)

        # construct valid mask, things you want is True
        valid_mask = torch.cat([
            torch.ones((tri_vpos_list.size(0), self.skip_token_num), dtype=torch.bool, device=valid_mask.device),
            valid_mask
        ], dim=1)

        return tri_vpos_list, valid_mask

    def construct_seq(self, tri_vpos_list, texture_patch_list, valid_mask, vns):
        """
        From input triangle list + texture patches, construct the sequence for transformer.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size], padded
        :param valid_mask: [batch_size, max_num_tri]
        :param vns: [batch_size, max_num_tri, 3, 3], padded
        :return: [batch_size, max_num_tri + 2, latent_dim]
        """
        batch_size = tri_vpos_list.size(0)

        # vertex normal encoding
        if self.config.use_vn_encoder:
            vn_emb = self.vn_encoder_norm(self.vn_encoding_proj(self.vn_pe(vns)))
        else:
            vn_emb = 0.

        # texture encoding
        B, N, C = texture_patch_list.shape
        size = self.config.texture_encode_patch_size
        
        # Build mask dynamically on the GPU
        tex_mask = torch.zeros((size, size), dtype=torch.bool, device=texture_patch_list.device)
        x, y = torch.meshgrid(torch.arange(size, device=texture_patch_list.device), 
                              torch.arange(size, device=texture_patch_list.device), indexing='ij')
        tex_mask[x + y <= size] = True
        
        # Efficient mathematically equivalent texture encoding to prevent OOM.
        # Instead of expanding [B, N, C] to [B, N, C, size, size] and masking it (which takes ~100GB of VRAM),
        # we mask and sum the Linear layer's weights spatially, then project [B, N, C] directly.
        weight = self.texture_encoder.weight
        bias = self.texture_encoder.bias
        
        latent_dim = weight.shape[0]
        # Reshape weight to [latent_dim, C, size, size]
        weight_reshaped = weight.view(latent_dim, C, size, size)
        
        # Mask the weights
        weight_masked = torch.where(
            tex_mask.unsqueeze(0).unsqueeze(0), 
            weight_reshaped, 
            torch.zeros_like(weight_reshaped)
        )
        
        # Sum over spatial dimensions to get [latent_dim, C]
        effective_weight = weight_masked.sum(dim=(2, 3))
        
        # Apply the linear projection directly
        tri_tex_emb = torch.nn.functional.linear(texture_patch_list, effective_weight, bias)
        tri_tex_emb = self.texture_encoder_norm(tri_tex_emb)

        # construct sequence
        tokens = []
        tokens.append(self.reg_tokens.expand(batch_size, -1, -1))

        if self.config.pe_type == 'nerf':
            tri_vpos_pe = self.tri_vpos_pe(tri_vpos_list)
            tri_emb = self.tri_encoding_norm(self.tri_encoding_proj(tri_vpos_pe)) + self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        elif self.config.pe_type == 'rope':
            tri_emb = self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        else:
            raise ValueError(f"Invalid positional encoding type: {self.config.pe_type}")

        seq = torch.cat(tokens, dim=1)

        # pad triangle pos (for RoPE) and valid mask (for all)
        # use center pos for RoPE on auxiliary tokens
        tri_vpos_list, valid_mask = self.process_tri_vpos_list(tri_vpos_list, valid_mask)

        return seq, valid_mask, tri_vpos_list

    def forward(self, tri_vpos_list, texture_patch_list, valid_mask, vns, rays_o, rays_d, tri_vpos_view_tf, tf32_view_tf=False):
        """
        Forward pass of the transformer.

        tri_vpos_list: [batch_size, max_num_tri, 9], padded
        texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size], padded
        valid_mask: [batch_size, max_num_tri], things you want is True
        vns: [batch_size, max_num_tri, 9], padded

        rays_o: [batch_size, num_views, 3]
        rays_d: [batch_size, num_views, img_h, img_w, 3]
        tri_vpos_view_tf: [batch_size, num_views, max_num_tri, 9], padded
        tf32_view_tf: bool, whether to use tf32 for view transformer
        """
        seq, valid_mask_padded, tri_vpos_list = self.construct_seq(tri_vpos_list, texture_patch_list, valid_mask, vns)
        seq = self.transformer(seq, src_key_padding_mask=valid_mask_padded, triangle_pos=tri_vpos_list)

        batch_size, num_views = rays_o.size(0), rays_o.size(1)
        seq = seq.repeat_interleave(num_views, dim=0)
        rays_o = rays_o.view(-1, *rays_o.shape[2:])
        rays_d = rays_d.view(-1, *rays_d.shape[2:])
        tri_vpos_view_tf = tri_vpos_view_tf.reshape(-1, *tri_vpos_view_tf.shape[2:])
        valid_mask = valid_mask.repeat_interleave(num_views, dim=0)
        valid_mask_padded = valid_mask_padded.repeat_interleave(num_views, dim=0)
        pos_seq, _ = self.process_tri_vpos_list(tri_vpos_view_tf, valid_mask)

        res = self.view_transformer(
            rays_o,
            rays_d,
            seq,
            pos_seq,
            valid_mask_padded,
            tf32_mode=tf32_view_tf
        )
        res = res.view(batch_size, num_views, *res.size()[1:])  # [batch_size * num_views, ...] -> [batch_size, num_views, ...]
        return res
