import torch
from torch import nn
import math

class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=True,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, q, k, v):
        N, L, C = q.shape
        M, L, C = k.shape
        # q = self.q(q).reshape(N, L, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        # k = self.v(k).reshape(M, L, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        # v = self.v(v).reshape(M, L, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        q = self.q(q).reshape(N, L*self.num_heads, C // self.num_heads).permute(1, 0, 2)
        k = self.v(k).reshape(M, L*self.num_heads, C // self.num_heads).permute(1, 0, 2)
        v1 = self.v(v).reshape(M, L, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = self.v(v).reshape(M, L*self.num_heads, C // self.num_heads).permute(1, 0, 2)
        if self.sparse_reg:
            attn = q * self.scale@ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn
            sparse = torch.mean(sparse, dim=2, keepdim=True)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v1)
        dconv_v = torch.mean(dconv_v, dim=2, keepdim=True)
        dconv_v = dconv_v.view(L*self.num_heads, 1, C // self.num_heads)
        attn = torch.bmm(k.transpose(-2, -1), v)
        v = torch.mean(v, dim=1, keepdim=True)
        if self.sparse_reg:
            x = (
                sparse@v
                + 0.5 * v
                + 1.0 / math.pi * q@attn
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * torch.bmm(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x = torch.bmm(q, attn)
        x += dconv_v
        x = x.transpose(1, 2)
        x = self.proj_drop(x)
        return x.view(N, L, C)
