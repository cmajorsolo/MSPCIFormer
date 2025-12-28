"""
Multi-Scale Patching Transformer with Channel Independence
"""

__all__ = ['MSPCIFormer']

from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from layers.MSPCILayer import MSPCILayer


class Model(nn.Module):
    """
    Multi-Scale patching Transformer without FFT: Channel-independent forecasting with predefined multi-scale patching

    Uses predefined patch sizes from scale_list_not_fft instead of FFT-based scale detection.
    Processes input at multiple scales and aggregates predictions.
    """

    def __init__(self, configs,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None, 
                 attn_dropout: float = 0.,
                 act: str = "gelu", 
                 pos_encoding: str = 'zeros',
                 learn_pe: bool = True, head_type='flatten',
                 **kwargs):

        super().__init__()

        # Load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout        
        head_dropout = configs.head_dropout        
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        self.scale_list = configs.scale_list_not_fft if hasattr(configs, 'scale_list_not_fft') else [4, 8, 16]
        self.top_k = configs.top_k if hasattr(configs, 'top_k') else len(self.scale_list)        
        self.selected_scales = self.scale_list[:self.top_k]        
        self.scale_models = nn.ModuleList([
            MSPCILayer(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,  # Different patch_len for each scale
                stride=stride,                
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,                
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                pos_encoding=pos_encoding,
                learn_pe=learn_pe,                
                head_dropout=head_dropout,
                padding_patch=padding_patch,                
                head_type=head_type,                
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,                
                **kwargs
            )
            for patch_len in self.selected_scales
        ])

        # Learnable weights for aggregating predictions from different scales
        self.scale_weights = nn.Parameter(torch.ones(1, self.top_k))

    def forward(self, x):        
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]        
        predictions = []
        for i, model in enumerate(self.scale_models):
            pred = model(x)  # [Batch, Channel, Output length]
            predictions.append(pred)
        # Stack predictions: [Batch, Channel, Output length, num_scales]
        predictions = torch.stack(predictions, dim=-1)
        # Apply learned weights and aggregate
        weights = F.softmax(self.scale_weights, dim=-1)  # [1, num_scales]
        weights = weights.view(1, 1, 1, -1)  # [1, 1, 1, num_scales]
        # Weighted sum across scales
        x = torch.sum(predictions * weights, dim=-1)  # [Batch, Channel, Output length]
        x = x.permute(0, 2, 1)  # x: [Batch, Output length, Channel]

        return x
