__all__ = ['MSPCILayer']

import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
from layers.PatchTST_layers import positional_encoding, get_activation_fn
from layers.RevIN import RevIN

class MSPCILayer(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,
                 n_layers:int=3, d_model=128, n_heads=16, d_ff:int=256, attn_dropout:float=0., 
                 dropout:float=0., act:str="gelu", pos_encoding:str='zeros', learn_pe:bool=True, 
                 head_dropout = 0, padding_patch = None, head_type = 'flatten', individual = False, 
                 revin = True, affine = True, subtract_last = False, **kwargs):

        super().__init__()        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1                
        self.ci_encoder = CIEncoder(c_in, patch_num=patch_num, patch_len=patch_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act,                                 
                                pos_encoding=pos_encoding, learn_pe=learn_pe, **kwargs)        
        self.head_nf = d_model * patch_num
        self.head = Flatten_Head(self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                  
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                        
        z = z.permute(0,1,3,2)                             
        z = self.ci_encoder(z)                                                                
        z = self.head(z)                                                                    
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z

class Flatten_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):        
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class CIEncoder(nn.Module): 
    def __init__(self, c_in, patch_num, patch_len,     
                 n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, attn_dropout=0., dropout=0., act="gelu",                
                 pos_encoding='zeros', learn_pe=True, **kwargs):       
        
        super().__init__()        
        self.patch_num = patch_num       
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pos_encoding, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(d_model, n_heads, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout, activation=act, n_layers=n_layers)

    def forward(self, x) -> Tensor:                                            
        n_vars = x.shape[1]
        x = x.permute(0,1,3,2)                                                  
        x = self.W_P(x)                                                          
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))     
        u = self.dropout(u + self.W_pos)          

        z = self.encoder(u)                                                      

        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                
        z = z.permute(0,1,3,2)                                                  
        
        return z    
            
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                    attn_dropout=0., dropout=0., activation='gelu', n_layers=1):
        super().__init__()        
        self.n_layers = n_layers
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.multi_head_attention = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=True), get_activation_fn(activation), nn.Dropout(dropout), nn.Linear(d_ff, d_model, bias=True))        
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src:Tensor):
        for i in range (self.n_layers):
            src_attn, attn = self.multi_head_attention(src, src, src)        
            src = src + self.dropout_attn(src_attn)        
            src_attn = self.ff(src)        
            src = src + self.dropout_ffn(src_attn)
        return src

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)        
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, lsa=lsa)        
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       

        output, attn_weights = self.sdp_attn(q_s, k_s, v_s)       
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)        
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)        

    def forward(self, q:Tensor, k:Tensor, v:Tensor):        
        attn_scores = torch.matmul(q, k) * self.scale   
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)        
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

