import torch
import torch.nn as nn
# from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg, _update_default_kwargs
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
# from .registry import register_pip_model
# from .registry import register_pip_model
from pathlib import Path
# from .grn import GlobalResponseNorm
# from .helpers import to_2tuple
from detectron2.layers import (
    CNNBlockBase,)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x



class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x

class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=3, agent_num=49, downstream_agent_shape=(7, 7), scale=-0.5, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** scale

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.kernel_size = kernel_size
        self.agent_num = agent_num

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                             padding=kernel_size // 2, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool_size = pool_size
        self.downstream_agent_shape = downstream_agent_shape
        self.pool = nn.AdaptiveAvgPool2d(output_size=downstream_agent_shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)
        H = int(n**0.5)
        W = int(n**0.5)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        # following code is under the condition of self.sr_ratio == 1
        assert self.sr_ratio == 1
        downstream_agent_num = self.downstream_agent_shape[0] * self.downstream_agent_shape[1]
        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, downstream_agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # interpolate hw
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(H, W), mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        position_bias1 = nn.functional.interpolate(position_bias1, size=self.downstream_agent_shape, mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, H * W, downstream_agent_num).permute(0, 2, 1)
        position_bias1 = position_bias1.reshape(1, num_heads, downstream_agent_num, H * W).repeat(b, 1, 1, 1)

        # interpolate hw
        position_bias2 = nn.functional.interpolate((self.ah_bias + self.aw_bias).squeeze(0), size=(H, W),
                                                   mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        position_bias2 = nn.functional.interpolate(position_bias2, size=self.downstream_agent_shape, mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, H * W, downstream_agent_num).permute(0, 2, 1)
        position_bias2 = position_bias2.reshape(1, num_heads, downstream_agent_num, H * W).repeat(b, 1, 1, 1)

        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # interpolate hw
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(H, W), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        agent_bias1 = nn.functional.interpolate(agent_bias1, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, H * W, downstream_agent_num).repeat(b, 1, 1, 1)

        # interpolate hw
        agent_bias2 = (self.ha_bias + self.wa_bias).squeeze(0).permute(0, 3, 1, 2)
        agent_bias2 = nn.functional.interpolate(agent_bias2, size=(H, W), mode='bilinear')
        agent_bias2 = agent_bias2.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        agent_bias2 = nn.functional.interpolate(agent_bias2, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias2 = agent_bias2.reshape(1, num_heads, H * W, downstream_agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H, W, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
# class Block(CNNBlockBase):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,):
        super().__init__()
        # super().__init__(in_channels, out_channels, stride)
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
            # self.mixer = AgentAttention(
            #     dim, num_patches=25,
            #     # dim, num_patches=64,
            #     num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            #     attn_drop=attn_drop, proj_drop=drop, sr_ratio=1,
            #     agent_num=49, downstream_agent_shape=(7, 7),
            #     kernel_size=3, scale=-0.5)
        else:
            self.mixer = MambaVisionMixer(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# res4_input = torch.rand(2, 1024, 14, 14).cuda()
# res4_depth = 6
# res5_depth = 3
# # B*1024*14*14
# # patch_embed = PatchEmbed(in_chans=3, in_dim=64, dim=256).cuda()
# # res4_input = patch_embed(res4_input)
# B, C, H, W = res4_input.shape
# x = window_partition(res4_input,14)
# # B*196*1024
#
# drop_path_res4=[0.067,0.083,0.100,0.117,0.133,0.150]
# res4_blocks = nn.ModuleList([
#     Block(dim=1024, counter=i, transformer_blocks=[3,4,5],
#           num_heads=8,
#           mlp_ratio=4,
#           qkv_bias=True,
#           qk_scale=None,
#           drop=0,
#           attn_drop=0,
#           drop_path=drop_path_res4[i] if isinstance(drop_path_res4, list) else drop_path_res4,
#           window_size=14,
#           layer_scale=None,
#           )
#     for i in range(res4_depth)]).cuda()
#
# for _, blk in enumerate(res4_blocks):
#     x = blk(x)
# res4_output = window_reverse(x, 14, H, W)   # B*1024*14*14
#
#
# # B*2048*7*7
# down_sample = Downsample(dim=1024).cuda()
# res5_input =down_sample(res4_output)
# B, C, H, W = res5_input.shape
# x1 = window_partition(res5_input,7)
# # B*49*2048
# drop_path_res5=[0.167,0.183,0.200]
# res5_blocks = nn.ModuleList([
#     Block(dim=2048, counter=i, transformer_blocks=[2],
#           num_heads=16,
#           mlp_ratio=4,
#           qkv_bias=True,
#           qk_scale=None,
#           drop=0,
#           attn_drop=0,
#           drop_path=drop_path_res5[i] if isinstance(drop_path_res5, list) else drop_path_res5,
#           window_size=7,
#           layer_scale=None,
#           )
#     for i in range(res5_depth)]).cuda()
#
# for _, blk in enumerate(res5_blocks):
#     x1 = blk(x1)
# res5_output = window_reverse(x1, 7, H, W)  # B*2024*7*7
#
# print(res5_output)
# print(res5_output.shape)

