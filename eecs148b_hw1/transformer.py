import torch
import numpy as np

class Linear(torch.nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
    std = np.sqrt(2 / (in_features + out_features))
    lim_std = 3
    torch.nn.init.trunc_normal_(self.weight, std=std, a=-lim_std*std, b=lim_std*std)
    
  def forward(self, x):
    return x @ self.weight.T
  
  
class Embedding(torch.nn.Module):
  def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    super().__init__()
    self.embeddings = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
    torch.nn.init.trunc_normal_(self.embeddings, std=1)
    
  def forward(self, token_ids: torch.Tensor):
    return self.embeddings[token_ids]
  
  
class LayerNorm(torch.nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()
    self.g = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
    self.b = torch.nn.Parameter(torch.zeros((d_model,), device=device, dtype=dtype))
    self.eps = eps
    
  def forward(self, x):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    x = (x - x.mean(dim=-1, keepdim=True)) / (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
    result = self.g * x + self.b
    return result.to(in_dtype)
  

class PositionwiseFeedForward(torch.nn.Module):
  def __init__(self, d_model, d_ff, device=None, dtype=None):
    super().__init__()
    self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
    self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
  def forward(self, x):
    h = self.linear1(x)
    return self.linear2(h * (h > 0))
  

class SinusoidalPositionalEncoding(torch.nn.Module):
  def __init__(self, d_model: int, max_seq_len: int, device=None, dtype=None):
    super().__init__()
    pe = torch.zeros((max_seq_len, d_model), device=device, dtype=dtype)
    positions = torch.arange(max_seq_len, device=device).unsqueeze(1) # (max_seq_len, 1)
    denom = torch.pow(10000, torch.arange(0, d_model, 2, device=device) / d_model).unsqueeze(0) # (1, d_model/2)
    pe[:, 0::2] = torch.sin(positions / denom)
    pe[:, 1::2] = torch.cos(positions / denom)
    self.register_buffer('pe', pe)
    
  def forward(self, token_positions: torch.Tensor):
    return self.pe[token_positions]
  
  
def softmax(x, dim=-1):
  x = x - torch.max(x, dim=dim, keepdim=True).values
  exp_x = torch.exp(x)
  return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def attention(q, k, v, mask=None):
  d_k = q.shape[-1]
  scores = q @ k.transpose(-2, -1) / np.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(~mask, float('-inf'))
  return softmax(scores, dim=-1) @ v


class MultiheadSelfAttention(torch.nn.Module):
  def __init__(self, d_model: int, num_heads: int):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    self.d_head = d_model // num_heads
    self.q_proj = Linear(d_model, d_model)
    self.k_proj = Linear(d_model, d_model)
    self.v_proj = Linear(d_model, d_model)
    self.o_proj = Linear(d_model, d_model)

  def forward(self, x):
    seq_len = x.shape[-2]
    # Project and reshape to (..., num_heads, seq_len, d_head)
    q = self.q_proj(x).unflatten(-1, (self.num_heads, self.d_head)).transpose(-2, -3)
    k = self.k_proj(x).unflatten(-1, (self.num_heads, self.d_head)).transpose(-2, -3)
    v = self.v_proj(x).unflatten(-1, (self.num_heads, self.d_head)).transpose(-2, -3)
    # Causal mask: (seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
    # Batched attention across all heads: (..., num_heads, seq_len, d_head)
    attn_out = attention(q, k, v, mask=causal_mask)
    # Reshape back to (..., seq_len, d_model)
    attn_out = attn_out.transpose(-2, -3).flatten(-2)
    return self.o_proj(attn_out)
  

class Transformer(torch.nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, use_layernorm: bool = True):
    super().__init__()
    self.layernorm1 = LayerNorm(d_model)
    self.mhsa = MultiheadSelfAttention(d_model, num_heads)
    self.layernorm2 = LayerNorm(d_model)
    self.ffn = PositionwiseFeedForward(d_model, d_ff)
    self.use_layernorm = use_layernorm
    
  def forward(self, x):
    if self.use_layernorm:
      x = x + self.mhsa(self.layernorm1(x))
      x = x + self.ffn(self.layernorm2(x))
    else:
      x = x + self.mhsa(x)
      x = x + self.ffn(x)
    return x
  
  
class TransformerLM(torch.nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, use_layernorm: bool = True, use_pe: bool = True):
    super().__init__()
    # token embeddings
    self.token_embedding = Embedding(vocab_size, d_model)
    # positional encodings
    self.positional_encoding = SinusoidalPositionalEncoding(d_model, context_length)
    self.use_pe = use_pe
    # transformer layers
    self.layers = torch.nn.ModuleList([Transformer(d_model, num_heads, d_ff, use_layernorm) for _ in range(num_layers)])
    # layernorm
    self.layernorm = LayerNorm(d_model)
    # LM head
    self.lm_head = Linear(d_model, vocab_size)
    
    
  def forward(self, token_ids):
    x = self.token_embedding(token_ids)
    if self.use_pe:
      x += self.positional_encoding(torch.arange(token_ids.shape[-1], device=token_ids.device))
    for layer in self.layers:
      x = layer(x)
    x = self.layernorm(x)
    return self.lm_head(x)