"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from ..core import register

__all__ = ["DFINESensorTransformer"]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
        # ---------------------------
        # Ablation switches (default ON)
        # ---------------------------
        enable_A=True,  # Box-in constraint
        enable_B=True,  # Structured anisotropic stretching
        enable_C=True,  # Spectral / SPD head-wise decoupling (orthogonal basis)
        # ---------------------------
        # Hyperparams
        # ---------------------------
        alpha=0.5,  # B: stretch strength for log(kappa)
        beta_lvl=0.3,  # B: per-level learnable tweak magnitude
        beta_box=1.0,  # A: tanh hardness for box-in (bigger => harder constraint)
        theta_range=math.pi / 4.0,  # C': limit head basis rotation to [-pi/4, pi/4]
    ):
        """
        Multi-Scale Deformable Attention (drop-in replacement)

        Innovations:
          (A) Box-in constraint:
              offsets -> tanh(beta_box * offsets), then map into reference box support
              => sampling points guaranteed near/inside the ref box (stable + interpretable)

          (B) Structured anisotropic stretching:
              use ref box aspect ratio to stretch offsets along major axis
              + per-level learnable elastic tweak (tanh), init identity

          (C') Spectral/SPD head-wise decoupling:
              each head learns an orthogonal basis U_h (2D rotation by theta_h)
              offsets are projected to U_h basis and projected back
              => head-specific directional subspace without changing norm (stable)

        Compatibility:
          - forward accepts extra args/kwargs (decoder may pass sensor_em)
          - supports reference_points last dim 2 or 4
          - automatically expand reference_points level dim from 1 to num_levels
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.method = method
        self.offset_scale = offset_scale

        # Ablation switches
        self.enable_A = enable_A
        self.enable_B = enable_B
        self.enable_C = enable_C

        # Hyperparams
        self.alpha = alpha
        self.beta_lvl = beta_lvl
        self.beta_box = beta_box
        self.theta_range = theta_range

        # num_points_list
        if isinstance(num_points, list):
            assert len(num_points) == num_levels, "len(num_points) must == num_levels"
            self.num_points_list = num_points
        else:
            self.num_points_list = [num_points for _ in range(num_levels)]

        # total points per head
        self.total_points_per_head = sum(self.num_points_list)  # P = ΣK_l
        self.total_points = self.num_heads * self.total_points_per_head

        # head dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # linear projections
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        # core attention
        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2,
            method=self.method,
        )

        # (B) per-level elastic anisotropy (init = 0 => identity)
        self.level_aniso = nn.Parameter(torch.zeros(self.num_levels, 2), requires_grad=True)  # [L,2]

        # (C') head basis parameter (angle), init = 0 => U = I
        self.head_theta = nn.Parameter(torch.zeros(self.num_heads), requires_grad=True)  # [H]

        # original style: scale factor for each point index (length = P)
        num_points_scale = []
        for n in self.num_points_list:
            num_points_scale += [1.0 / n for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        # debug
        self.debug = False
        self.debug_data = None

        self._reset_parameters()

        # discrete mode keeps offsets fixed
        if self.method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    # ---------------------------
    # Public APIs for ablation
    # ---------------------------
    def set_ablation(self, A=None, B=None, C=None):
        """Set module-level ablation switches at runtime."""
        if A is not None:
            self.enable_A = bool(A)
        if B is not None:
            self.enable_B = bool(B)
        if C is not None:
            self.enable_C = bool(C)

    def _reset_parameters(self):
        # sampling_offsets init (same spirit as original)
        init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [H,2]
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values

        # [H,1,2] -> [H,P,2]
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.total_points_per_head, 1)

        # radius grows with point index
        scaling = torch.arange(1, self.total_points_per_head + 1, dtype=torch.float32).view(1, -1, 1)
        grid_init = grid_init * scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights init
        init.constant_(self.attention_weights.weight, 0.0)
        init.constant_(self.attention_weights.bias, 0.0)

        # new params: default no effect
        nn.init.constant_(self.level_aniso, 0.0)
        nn.init.constant_(self.head_theta, 0.0)

    @staticmethod
    def _as_tensor_spatial_shapes(value_spatial_shapes, dtype, device):
        if torch.is_tensor(value_spatial_shapes):
            return value_spatial_shapes.to(dtype=dtype, device=device)
        return torch.as_tensor(value_spatial_shapes, dtype=dtype, device=device)

    def _build_head_basis(self, device, dtype):
        """
        Build per-head orthogonal basis U_h (2D rotation).
        theta_h is constrained to [-theta_range, theta_range] for stability.

        Return:
          U:   [H,2,2]
          U_t: [H,2,2]
        """
        theta = torch.tanh(self.head_theta.to(device=device, dtype=dtype)) * self.theta_range  # [H]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        U = torch.stack(
            [
                torch.stack([cos_t, -sin_t], dim=-1),
                torch.stack([sin_t, cos_t], dim=-1),
            ],
            dim=-2,
        )  # [H,2,2]
        U_t = U.transpose(-1, -2)
        return U, U_t

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[int],
        *args,
        **kwargs,
    ):
        """
        Args:
          query: [B, Q, C]
          reference_points:
            - [B, Q, L, 2] OR
            - [B, Q, L, 4] (cx, cy, w, h)
            decoder 常见是 [B,Q,1,4]，这里会自动 expand 到 [B,Q,num_levels,4]
          value: [B, V, C]
          value_spatial_shapes: [L,2]
        """
        B, Q = query.shape[:2]
        H = self.num_heads
        L = self.num_levels
        P = self.total_points_per_head

        # 1) base offsets & weights
        sampling_offsets = self.sampling_offsets(query).view(B, Q, H, P, 2)  # [B,Q,H,P,2]
        attention_weights = self.attention_weights(query).view(B, Q, H, P)  # [B,Q,H,P]
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 2) fix ref level dim
        if reference_points.dim() != 4:
            raise ValueError(f"reference_points must be 4D, got {reference_points.shape}")
        if reference_points.shape[2] == 1 and L > 1:
            reference_points = reference_points.expand(B, Q, L, reference_points.shape[-1]).contiguous()

        # =========================================================
        # (B) Structured anisotropic stretching (optional)
        # only active when ref last dim == 4
        # =========================================================
        if self.enable_B and reference_points.shape[-1] == 4:
            wh = reference_points[..., 2:4]  # [B,Q,L,2]
            w_ = wh[..., 0]
            h_ = wh[..., 1]
            eps = 1e-6
            max_wh = torch.max(w_, h_)
            min_wh = torch.clamp(torch.min(w_, h_), min=eps)
            kappa = max_wh / min_wh  # [B,Q,L]

            # non-saturating stretch: 1 + alpha*log(kappa)
            base_stretch = 1.0 + self.alpha * torch.log(torch.clamp(kappa, min=1.0))  # [B,Q,L]
            axis_mask_x = (w_ >= h_).to(query.dtype)
            axis_mask_y = 1.0 - axis_mask_x
            stretch_x = axis_mask_x * base_stretch + axis_mask_y * 1.0
            stretch_y = axis_mask_y * base_stretch + axis_mask_x * 1.0

            # per-level learnable tweak (init identity)
            gamma = torch.tanh(self.level_aniso.to(dtype=query.dtype, device=query.device)) * self.beta_lvl  # [L,2]
            level_scale_x = (1.0 + gamma[:, 0]).view(1, 1, L)
            level_scale_y = (1.0 + gamma[:, 1]).view(1, 1, L)
            stretch_x = stretch_x * level_scale_x
            stretch_y = stretch_y * level_scale_y

            # apply stretch per level slice
            new_offsets = []
            start = 0
            for lvl, K in enumerate(self.num_points_list):
                end = start + K
                off_l = sampling_offsets[:, :, :, start:end, :]  # [B,Q,H,K,2]
                sx = stretch_x[:, :, lvl].view(B, Q, 1, 1)
                sy = stretch_y[:, :, lvl].view(B, Q, 1, 1)
                dx = off_l[..., 0] * sx
                dy = off_l[..., 1] * sy
                new_offsets.append(torch.stack([dx, dy], dim=-1))
                start = end
            sampling_offsets = torch.cat(new_offsets, dim=3)  # [B,Q,H,P,2]

        # =========================================================
        # (C') Spectral/SPD head-wise decoupling (optional)
        # off' = U * (U^T * off) (pure basis change)
        # =========================================================
        if self.enable_C and reference_points.shape[-1] == 4:
            U, U_t = self._build_head_basis(device=query.device, dtype=query.dtype)  # [H,2,2]

            new_offsets = []
            start = 0
            for lvl, K in enumerate(self.num_points_list):
                end = start + K
                off_l = sampling_offsets[:, :, :, start:end, :]  # [B,Q,H,K,2]
                x = torch.einsum("bqhkf,hfg->bqhkg", off_l, U_t)
                off_p = torch.einsum("bqhkf,hfg->bqhkg", x, U)
                new_offsets.append(off_p)
                start = end
            sampling_offsets = torch.cat(new_offsets, dim=3)  # [B,Q,H,P,2]

        # =========================================================
        # 3) sampling locations
        # =========================================================
        if reference_points.shape[-1] == 2:
            # encoder-style ref points
            spatial_shapes = self._as_tensor_spatial_shapes(value_spatial_shapes, query.dtype, query.device)  # [L,2]
            normalizer = spatial_shapes.flip([1])  # [L,2] (W,H)

            loc_list = []
            start = 0
            for lvl, K in enumerate(self.num_points_list):
                end = start + K
                off_l = sampling_offsets[:, :, :, start:end, :]  # [B,Q,H,K,2]
                ref_xy = reference_points[:, :, lvl, :].view(B, Q, 1, 1, 2).expand(B, Q, H, K, 2)
                norm = normalizer[lvl].view(1, 1, 1, 1, 2)
                loc_l = ref_xy + off_l / norm
                loc_list.append(loc_l)
                start = end
            sampling_locations = torch.cat(loc_list, dim=3)  # [B,Q,H,P,2]

        elif reference_points.shape[-1] == 4:
            # decoder-style ref box (cx,cy,w,h)
            num_points_scale = self.num_points_scale.to(dtype=query.dtype, device=query.device)  # [P]

            loc_list = []
            start = 0
            for lvl, K in enumerate(self.num_points_list):
                end = start + K
                off_l = sampling_offsets[:, :, :, start:end, :]  # [B,Q,H,K,2]
                s_l = num_points_scale[start:end].view(1, 1, 1, K, 1)
                ref_ctr = reference_points[:, :, lvl, 0:2].view(B, Q, 1, 1, 2)
                ref_wh = reference_points[:, :, lvl, 2:4].view(B, Q, 1, 1, 2)

                off_scaled = off_l * s_l * self.offset_scale  # [B,Q,H,K,2]

                if self.enable_A:
                    off_unit = torch.tanh(self.beta_box * off_scaled)
                    offset_l = off_unit * (ref_wh * 0.5)
                else:
                    offset_l = off_scaled * ref_wh

                loc_l = ref_ctr + offset_l
                loc_list.append(loc_l)
                start = end

            sampling_locations = torch.cat(loc_list, dim=3)  # [B,Q,H,P,2]

        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        # debug cache
        if self.debug:
            self.debug_data = {
                "sampling_locations": sampling_locations.detach(),
                "attention_weights": attention_weights.detach(),
                "reference_points": reference_points.detach(),
                "sampling_offsets": sampling_offsets.detach(),
            }

        # 4) core attention
        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )
        return output


class HighFreqSensorEncoder(nn.Module):
    def __init__(self, use_heading=False, num_freqs=4, hidden_dim=128):
        """
        Args:
          use_heading (bool): 是否用航向角
          num_freqs (int): 多少个频率阶数，越大越细腻
          hidden_dim (int): 中间层特征维度
        """
        super().__init__()
        self.use_heading = use_heading
        self.num_freqs = num_freqs

        if use_heading:
            base_input_dim = 3  # height, pitch, heading
        else:
            base_input_dim = 2  # height, pitch

        # 每个 scalar 都展开成 sin/cos * num_freqs * 2
        self.total_input_dim = base_input_dim * num_freqs * 2

        self.mlp = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _encode_scalar(self, x, freq_bands):
        """
        对每个scalar做高频sin-cos编码
        Args:
          x: Tensor [B,]
          freq_bands: Tensor [num_freqs,]
        Returns:
          Tensor [B, num_freqs * 2]
        """
        x = x.unsqueeze(-1) * freq_bands.unsqueeze(0)  # [B, num_freqs]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def forward(self, sensor_feat_list=None):
        """
        Args:
          sensor_feat_list: Tensor [B, 3] (height, pitch_deg, heading_deg)
        Returns:
          Tensor: [B, hidden_dim]
        """
        height = sensor_feat_list[:, 0]
        pitch_deg = sensor_feat_list[:, 1]
        heading_deg = sensor_feat_list[:, 2]

        # 归一化高度
        height_norm = (height - 5.0) / (260.0 - 5.0)
        height_norm = torch.clamp(height_norm, 0.0, 1.0)

        # pitch角转弧度
        pitch_rad = pitch_deg / 180.0 * math.pi

        # heading角转弧度（可选）
        if self.use_heading:
            assert heading_deg is not None
            heading_rad = heading_deg / 180.0 * math.pi

        # 构造频率
        freq_bands = 2 ** torch.linspace(
            0,
            self.num_freqs - 1,
            self.num_freqs,
            device=height.device,
        ) * math.pi

        height_embed = self._encode_scalar(height_norm, freq_bands)
        pitch_embed = self._encode_scalar(pitch_rad, freq_bands)

        if self.use_heading:
            heading_embed = self._encode_scalar(heading_rad, freq_bands)
            sensor_embed = torch.cat([height_embed, pitch_embed, heading_embed], dim=-1)
        else:
            sensor_embed = torch.cat([height_embed, pitch_embed], dim=-1)

        out = self.mlp(sensor_embed)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
    ):
        super().__init__()

        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
        )
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )
        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))
        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.
    This layer computes the target location using the formula:
      sum{Pr(n) * W(n)}
    where Pr(n) is the softmax probability vector representing the discrete distribution,
    and W(n) is the non-uniform Weighting Function.

    Args:
      reg_max (int): Max number of the discrete bins. Default is 32.
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).
    """

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act="relu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]
        )

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """Preprocess values for MSDeformableAttention."""
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        iou_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0

        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_quality = []

        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])

            output_detach = output.detach()
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(
                ref_points_initial,
                integral(pred_corners, project),
                reg_scale,
            )

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                quality = torch.sigmoid(iou_head[i](output))  # [B,Q,1]
                dec_out_quality.append(quality)

            if (not self.training) and (i >= self.eval_idx):
                break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
            torch.stack(dec_out_quality),
        )


@register()
class DFINESensorTransformer(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
    ):
        super().__init__()

        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ("default", "one2many", "agnostic"), ""
        assert cross_attn_method in ("default", "discrete"), ""
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module (sensor)
        self.sensor_encoder = HighFreqSensorEncoder(use_heading=False, num_freqs=4, hidden_dim=hidden_dim)
        self.sensorscale = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.sensor_delta_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sensor_query_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
        )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    ("norm", nn.LayerNorm(hidden_dim)),
                ]
            )
        )
        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        # decoder head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
            + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.integral = Integral(self.reg_max)

        self.dec_iou_head = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, 1) for _ in range(num_layers - self.eval_idx - 1)]
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        # (原代码里这里重复了一次，保留你逻辑但实际上不需要第二次)
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        init.constant_(self.sensor_delta_proj.weight, 0)
        init.constant_(self.sensor_delta_proj.bias, 0)
        init.constant_(self.sensor_query_gate.weight, 0)
        init.constant_(self.sensor_query_gate.bias, -2.0)

        for h in self.dec_iou_head:
            init.constant_(h.weight, 0)
            init.constant_(h.bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)

        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))  # [b, h*w, c]
            spatial_shapes.append([h, w])  # [num_levels, 2]

        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory: torch.Tensor, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory: torch.Tensor = self.enc_output(memory)

        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        else:
            raise ValueError(f"Unknown query_select_method: {self.query_select_method}")

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
        )
        topk_logits = (
            outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]),
            )
            if self.training
            else None
        )
        topk_memory = memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]),
        )
        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None, sensor_feats=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=1.0,
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        if sensor_feats is not None:
            sensor_feats = sensor_feats.to(device=init_ref_contents.device, dtype=init_ref_contents.dtype)
            valid = (sensor_feats.abs().sum(dim=1, keepdim=True) > 1e-6).to(init_ref_contents.dtype)
            valid = valid.unsqueeze(-1)

            sensor_embedding = self.sensor_encoder(sensor_feats)
            sensor_embedding = self.sensorscale * sensor_embedding
            delta = self.sensor_delta_proj(sensor_embedding).unsqueeze(1)  # [B,1,C]

            B, Q, C = init_ref_contents.shape
            sensor_expand = sensor_embedding.unsqueeze(1).expand(B, Q, C)
            gate_in = torch.cat([init_ref_contents, sensor_expand], dim=-1)  # [B,Q,2C]
            gate = torch.sigmoid(self.sensor_query_gate(gate_in))

            init_ref_contents = init_ref_contents + (gate * delta) * valid

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits, out_quality = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_iou_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
            dn_out_quality, out_quality = torch.split(out_quality, dn_meta["dn_num_split"], dim=2)

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_quality": out_quality[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_quality": out_quality[-1],
            }

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
                outputs_quality=out_quality[:-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                    outputs_quality=dn_out_quality[:-1],
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
        outputs_quality=None,
    ):
        if outputs_quality is None:
            return [
                {
                    "pred_logits": a,
                    "pred_boxes": b,
                    "pred_corners": c,
                    "ref_points": d,
                    "teacher_corners": teacher_corners,
                    "teacher_logits": teacher_logits,
                }
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
            ]
        else:
            return [
                {
                    "pred_logits": a,
                    "pred_boxes": b,
                    "pred_corners": c,
                    "ref_points": d,
                    "pred_quality": q,
                    "teacher_corners": teacher_corners,
                    "teacher_logits": teacher_logits,
                }
                for a, b, c, d, q in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref, outputs_quality)
            ]
