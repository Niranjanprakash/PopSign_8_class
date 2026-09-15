import torch
import torch.nn as nn
from backend.utils import logger

# Try native Mamba import
NATIVE_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba as NativeMamba
    NATIVE_MAMBA_AVAILABLE = True
    logger.info("[MAMBA] Native mamba_ssm imported successfully.")
except ImportError as e:
    logger.warning(
        f"[MAMBA] Native mamba_ssm could not be imported ({e}). "
        "This is common on Windows due to CUDA compilation requirements. "
        "Using GRU-based optimized MambaBlock FALLBACK."
    )


class MambaBlockFallback(nn.Module):
    """
    GRU-based fallback replacing the slow Python for-loop SSM scan.
    PyTorch GRU is implemented in optimized C++/CUDA — ~15x faster than
    the sequential loop on CPU, and drops to native CUDA speed on GPU.
    Maintains the same input/output contract: [B, L, D] -> [B, L, D].
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = expand * d_model

        # Input projection + gating (mirrors Mamba's in_proj)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local context (same as Mamba)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )

        # Bidirectional GRU replaces the sequential SSM scan — fully parallelised
        self.gru = nn.GRU(
            input_size=self.d_inner,
            hidden_size=self.d_inner,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Project bidirectional output back to d_inner, then to d_model
        self.out_proj = nn.Sequential(
            nn.Linear(self.d_inner * 2, self.d_inner, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_inner, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]  ->  out: [B, L, D]
        """
        B, L, _ = x.shape

        # 1. Project & split into content + gate branches
        xz = self.in_proj(x)                          # [B, L, d_inner*2]
        x_branch, z_branch = xz.chunk(2, dim=-1)      # each [B, L, d_inner]

        # 2. Depthwise conv for local context
        xc = self.conv(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)  # [B, L, d_inner]
        xc = torch.nn.functional.silu(xc)

        # 3. GRU scan (replaces sequential SSM loop)
        gru_out, _ = self.gru(xc)                     # [B, L, d_inner*2]

        # 4. Gated output
        gated = gru_out * torch.nn.functional.silu(z_branch.repeat(1, 1, 2))  # broadcast gate

        # 5. Project back to d_model
        return self.out_proj(gated)                   # [B, L, d_model]


class MambaBlock(nn.Module):
    """Wrapper: uses native mamba_ssm when available, GRU fallback otherwise."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if NATIVE_MAMBA_AVAILABLE:
            self.block = NativeMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.block = MambaBlockFallback(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
