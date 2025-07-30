import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as TF
import numpy as np
import math
from torch.utils.data import Dataset,DataLoader
from typing import Optional

from target import ImagePatchesDataset

"""
Supposedly Target_Encoder components for I-Jepa Target images.

"""

class PatchEmbedding(nn.Module):
    """
    Learns a linear projection of flattened image patches into embedding space.

    Input:
        x [B, patch_dim, num_patches]
    Output:
        Tensor [B, num_patches, embed_dim]
    """
    def __init__(self, patch_dim, embed_dim):
        super().__init__()

        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):

        # Convert to [B, num_patches, patch_dim]
        x = x.permute(0, 2, 1)
        # Project to embedding dim: [B, num_patches, embed_dim]
        x = self.proj(x)
        return x

class PositionalEncoding(nn.Module):
    """
    sinusoidal positional embeddings.No CLS token prepended as I-JEPA paper didnt use.
    """
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        pos_in = torch.zeros(max_len, embed_dim)
        
        # shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # This term ensures that wavelengths vary geometrically.
        # based on 1 / (10000^(2i / embed_dim))
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * - (math.log(10000.0) / embed_dim)
        )
        
        pos_in[:, 0::2] = torch.sin(position * div_term)
        pos_in[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension [1, max_len, embed_dim]
        pos_in = pos_in.unsqueeze(0) 
        self.register_buffer('pos_embed', pos_in)

    def forward(self, x):
        N = x.size(1)
        
        if N > self.pos_embed.size(1): #type:ignore
            raise ValueError(f"Sequence length {N} exceeds max_len {self.pos_embed.size(1)} specified during initialization.") #type:ignore

        x = x + self.pos_embed[:, :N] #type:ignore
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learned projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout & LayerNorm (for residual + norm)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    #matrix multiply attention weights by value vectors
    def max_multi_v(self, attn_weights, value_tensor):
        # attn_weights: (batch, heads, seq_len, seq_len)
        # value_tensor: (batch, heads, seq_len, head_dim)
        return torch.matmul(attn_weights, value_tensor)

    #concatenate heads
    def concat_attention_heads(self, tensor):
        # tensor: (batch, heads, seq_len, head_dim)
        b, h, t, d = tensor.size()
        #(batch, seq_len, heads*head_dim)
        return tensor.transpose(1, 2).contiguous().view(b, t, h * d)

    #apply final linear transform to concatenated heads
    def linear_transform_concatenated_head(self, tensor):
        #(batch, seq_len, embed_dim)
        return self.out_proj(tensor)

    def forward(self, query, key=None, value=None, mask=None):
        """
        query,key,value: (batch_size, seq_len, embed_dim)
        mask: (batch_size, 1, seq_len, seq_len) | None
        """

        if key is None:   key = query
        if value is None: value = query

        b, t, _ = query.size()

        #Project inputs
        Q = self.q_proj(query)  # (b, t, embed_dim)
        K = self.k_proj(key)
        V = self.v_proj(value)

        #Reshape for multiple heads: (b, heads, t, head_dim)
        Q = Q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        #Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = TF.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #Weighted sum
        attn_output = self.max_multi_v(attn_weights, V)  # (b, heads, t, head_dim)

        #Concatenate heads
        concat_output = self.concat_attention_heads(attn_output)  # (b, t, embed_dim)

        #Final linear projection + dropout
        proj_output = self.linear_transform_concatenated_head(concat_output)
        proj_output = self.dropout(proj_output)

        #Residual + LayerNorm
        output = self.layernorm(query + proj_output)

        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_hidden_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        ff_out = self.net(x)
        # residual + norm
        return self.layernorm(x + ff_out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim, dropout)

    def forward(self, x, mask=None):
        #self‑attention
        x, _ = self.self_attn(x, x, x, mask)
        #feed‑forward
        x = self.feed_forward(x)
        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(self, image_size :int, patch_size :int, in_channels : int, embed_dim :int, num_heads :int, ff_hidden_dim :int, num_layers :int, dropout :Optional[float] = 0.1):
        super().__init__()
        
        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2
        # Calculate patch_dim
        patch_dim = in_channels * (patch_size ** 2)
        self.patch_embed_layer = PatchEmbedding(patch_dim, embed_dim)
        self.pos_embed_layer = PositionalEncoding(embed_dim, max_len=num_patches)
        
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed_layer(x)
        x = self.pos_embed_layer(x) 
        x = self.dropout(x)

        # encoder
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        return x


if __name__ == '__main__':
    init_data = ImagePatchesDataset(r"D:\Image_Jepa\src\utils\photos_no_class")
    data_load = DataLoader(init_data, batch_size= 4, shuffle= True)
    for batch in data_load:
        img_batch = batch
        break
    # init_image_embed = PatchEmbedding(768, 64)
    # check_embed = init_image_embed.forward(img_batch)
    # pos_embed = PositionalEncoding(64, 200)
    # print(pos_embed.forward(check_embed).size())
    # print(check_embed.size())
    vision_encode = VisionTransformerEncoder(224, 16, 3, 768, 16, 512, 6)
    print(vision_encode.forward(img_batch))
