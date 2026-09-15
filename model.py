import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from dataclasses import dataclass
import math 
from typing import Optional 

@dataclass 
class ModelArgs:
    dim : int = 4096 
    n_layers : int = 32
    n_heads : int = 32
    n_kv_heads : Optional[int] = None
    vocab_size : int = -1 # later set in the build method 
    multiple_of : int = 256 # the dimension should be a multiple of this value
    ffn_dim_multiplier : Optional[float] = None # the multiplier for the feed-forward network dimension
    norm_eps : float = 1e-5 # the epsilon value for layer normalization

    # needed for KV cache 
    max_batch_size : int = 32 
    max_seq_len : int = 2048 # the maximum sequence length for the KV cache
    device : str = None 

class RMSNorm(nn.Module):
    """ 
    Root Mean Square Layer Normalization (RMSNorm) implementation.
    This normalization technique normalizes the input based on the root mean square of the elements.
    """
    def __init__(self, dim : int , eps : float = 1e-6):
        super().__init__()
        self.eps = eps # the epsilon value for numerical stability
        # the gamma parameter 
        self.weight = nn.Parameter(torch.ones(dim)) # the learnable weight parameter for RMSNorm

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # (B, Seqlen, dim) * (B , seqlen, 1) = (B, Seqlen, dim)
        # rsqrt  = 1/ sqrt(mean(x^2) + eps)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


def precompute_theta_pos_frequencies(head_dim : int,
                                     seq_len : int, 
                                     device : str, 
                                     theta : float = 10000.0):
    """ 
    Precompute the positional frequencies for rotary positional embeddings.

    Args:
        head_dim (int): The dimension of each attention head.
        seq_len (int): The maximum sequence length.
        device (str): The device to store the computed frequencies.
        theta (float, optional): The base frequency. Default is 10000.0.

    Returns:
        torch.Tensor: The precomputed positional frequencies of shape (seq_len, head_dim).
    """
    assert head_dim % 2 == 0, "head_dim must be even for rotary positional embeddings."
    # build the theta parameter for each dimension
    # according to the formula theta_i = 10000^(-2(i-1)/head_dim)
    # shape : (head_dim // 2,)
    theta_numerator = torch.arange(0, head_dim, 2).float() # the indices for the even dimensions
    # shape : (head_dim // 2,)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # shape : (head_dim // 2,) 
    # theta computes the inverse frequencies for the rotary positional embeddings
    # construct the positions  the "m" parameter 
    # shape : (seq_len) 
    m = torch.arange(seq_len, device=device) # shape : (seq_len) 
    # muliply each theta by each position using the outer product
    # shape : (seq_len) outer product * (head_dim // 2) = (seq_len, head_dim // 2)
    freqs = torch.outer(m, theta).float() # shape : (seq_len, head_dim // 2)
    # convert the frequencies to complex numbers using polar coordinates  , where polar means (magnitude, angle) = (1, frequency)
    freq_complex =  torch.polar(torch.ones_like(freqs), freqs) # shape : (seq_len, head_dim // 2)
    return freq_complex

def apply_rotary_embeddings(x : torch.Tensor, 
                            freqs_complex: torch.Tensor, device : str) -> torch.Tensor:
    """ 
    Apply rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_heads, head_dim).
        freqs_complex (torch.Tensor): The precomputed complex frequencies of shape (seq_len, head_dim // 2).
        device (str): The device to perform the computation on.

    Returns:
        torch.Tensor: The input tensor with rotary positional embeddings applied, of the same shape as `x`.
    """
    # separate the last dimension paris of two vlues , representing the real and the imaginary parts of the complex number 
    # two consecutive values will become a single complex number representing the rotary embedding.
    # (B,seq_len,num_heads,head_dim) -> (B,seq_len,num_heads,head_dim // 2) complex
    B, seq_len, num_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # view_as_complex to convert the last dimension pairs into complex numbers representing the rotary embeddings
    # reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension.
    # (Seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # add batch and head dimensions where unsqueeze is used to match the shape of x_complex unsqueeze(0) means adding a new dimension at the 0th position (batch dimension) and unsqueeze(2) means adding a new dimension at the 2nd position (head dimension)
    # multiply the each complex number in x_complex by the corresponding complex number in the freqs complex tensor
    # results in rotation of the complex number in the complex plane
    x_rotated = x_complex * freqs_complex
    # convert back to real tensor
    x_rotated_real = torch.view_as_real(x_rotated)
    x_out = x_rotated_real.reshape(*x.shape) # B,seqlen,H,Head_dim/2, 2 
    return x_out

"""" 
Rotary positional embeddings Example math oprtation: 
    Given a complex number x = a + bi and a complex frequency f = cos(theta) + i*sin(theta),
    the rotary embedding is applied as:
        x_rotated = x * f
    which results in:
        x_rotated = (a + bi) * (cos(theta) + i*sin(theta))
                  = (a*cos(theta) - b*sin(theta)) + (a*sin(theta) + b*cos(theta))i
    This corresponds to a rotation of the complex number x in the complex plane by the angle theta.

    In the context of rotary positional embeddings in transformers, this rotation is applied to the query and key vectors in the attention mechanism, allowing the model to encode relative positional information effectively.
Got it — let’s rewrite the RoPE example in **plain text with matrices**, no LaTeX.

---

### Core Idea
RoPE rotates pairs of dimensions in the query and key vectors.  
For each pair `(q_2i, q_2i+1)`, you apply a 2D rotation matrix that depends on the token’s position.

The rotation looks like this:

```
[q_2i'     ]   [ cos(θ)  -sin(θ) ] [ q_2i     ]
[q_2i+1'   ] = [ sin(θ)   cos(θ) ] [ q_2i+1   ]
```

Same thing happens for the key vector.

---

### Example with 2D Vectors
Let’s say:

- Query at position p=2 → q = [1, 2]  
- Key at position p=3 → k = [2, 1]  
- Rotation angle θ = 45° (π/4)

Rotation matrix:

```
R(θ) = [ 0.707  -0.707 ]
       [ 0.707   0.707 ]
```

Rotate query:

```
q' = R * q
   = [ 0.707*1 + (-0.707)*2 ]
     [ 0.707*1 +  0.707*2   ]
   = [ -0.707,  2.121 ]
```

Rotate key:

```
k' = R * k
   = [ 0.707*2 + (-0.707)*1 ]
     [ 0.707*2 +  0.707*1   ]
   = [ 0.707,  2.121 ]
```

Dot product:

```
q' · k' = (-0.707 * 0.707) + (2.121 * 2.121)
        ≈ 4.0
```

---

### Example with 4D Vectors
Now let’s extend to 4D.  
Query vector: q = [1, 2, 3, 4]  
Angles: θ₀ = 45° (for first pair), θ₁ = 30° (for second pair)

Block diagonal rotation matrix:

```
R_p = [ 0.707  -0.707   0      0    ]
      [ 0.707   0.707   0      0    ]
      [ 0       0      0.866  -0.5  ]
      [ 0       0      0.5     0.866]
```

Rotate q:

- First pair [1, 2] → [-0.707, 2.121]  
- Second pair [3, 4] → [0.598, 4.964]  

So rotated query:

```
q' = [-0.707, 2.121, 0.598, 4.964]
```

---

✅ Takeaway: In higher dimensions, RoPE applies **independent 2D rotations** to each consecutive pair of values. The full rotation matrix is block diagonal, and the dot product of rotated queries and keys naturally encodes relative positions.

---

"""


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat the key or value tensor `n_rep` times along the head dimension.

    Purpose:
        This function repeats the key or value tensor along the head dimension to match the number of query heads.
    Example:
        x has shape (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep = 2
        The output will have shape (batch_size, seq_len, n_kv_heads * 2, head_dim)
    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_len, n_kv_heads, head_dim).
        n_rep (int): The number of times to repeat the tensor along the head dimension.

    Returns:
        torch.Tensor: The repeated tensor of shape (batch_size, seq_len, n_kv_heads * n_rep, head_dim).
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim) 1 represents the new dimension for repetition
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )



class SelfAttention(nn.Module):
    """
    Self-attention module with support for rotary positional embeddings and key-value head repetition.
    This module handles the computation of self-attention, including the repetition of key and value heads to match the number of query heads and the application of rotary positional embeddings.
    It is designed to work efficiently with large sequences by caching key and value tensors and supports flexible head configurations.
    Args:
        args (ModelArgs): The model arguments containing configuration for the self-attention module.
        n_kv_heads (int, optional): The number of key-value heads. If None, it defaults to the number of query heads.
        n_heads_q (int): The number of query heads.
        n_rep (int): The number of times the key-value heads should be repeated to match the query heads.
        head_dim (int): The dimension of each attention head.
        cache_k (torch.Tensor): The cache for key tensors.
        cache_v (torch.Tensor): The cache for value tensors.
        start_pos (int): The starting position for the current sequence in the cache.
        freqs_complex (torch.Tensor): The precomputed rotary positional embeddings.
        x (torch.Tensor): The input tensor containing the query sequence.
    Returns:
        torch.Tensor: The output tensor after applying self-attention.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)


class FeedForward(nn.Module):
    """
    Feed-forward neural network module used within the transformer architecture.

    This module typically consists of two linear layers with an activation function in between.
    It is applied independently to each position in the sequence.

    Args:
        dim (int): The input and output dimension of the feed-forward network.
        hidden_dim (int): The hidden layer dimension.

    Returns:
        torch.Tensor: The output tensor after applying the feed-forward network.
    """

    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim # Initial expansion factor for the hidden dimension in the feed-forward network
        hidden_dim = int(2 * hidden_dim / 3) # Reduce the hidden dimension by a factor of 2/3
        if args.ffn_dim_multiplier is not None: # Apply the feed-forward network dimension multiplier if specified
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter for efficient computation
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    """
    A single encoder block within the transformer architecture.

    This block typically consists of a multi-head self-attention mechanism followed by a feed-forward neural network.
    Layer normalization and residual connections are applied around both the attention and feed-forward sub-layers.

    Args:
        args (ModelArgs): The model arguments containing configuration parameters.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads # Dimension of each attention head

        # normalization layers for the attention and feed-forward sub-layers
        self.attention_norm = RMSNorm(args.dim, eps= args.norm_eps) # Root mean square normalization for the attention sub-layer
        self.feed_forward_norm = RMSNorm(args.dim, eps= args.norm_eps) # Root mean square normalization for the feed-forward sub-layer

    def forward(self, x : torch.Tensor, start_ops: int, freqs_complex : torch.Tensor):  
        # (B, Seqlen,Dim) + (B, seqlen, Dim) --> (B, Seqlen, Dim) after attention and feed-forward operations
        h = x + self.attention.forward(
            self.attention_norm(x),
            start_ops=start_ops,
            freqs_complex=freqs_complex
        )
        x = h + self.feed_forward(self.feed_forward_norm(h))
        return x

class Transformer(nn.Module):
    """
    The full transformer model consisting of multiple encoder blocks.

    Args:
        args (ModelArgs): The model arguments containing configuration parameters.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args 
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList() # List of encoder blocks
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args)) # Append an encoder block to the list of layers

        self.norm = RMSNorm(args.dim, eps=args.norm_eps) # Root mean square normalization for the final output of the transformer   
        self.output_proj = nn.Linear(args.dim, self.vocab_size,bias=False) # Linear projection to the vocabulary size for output logits , bias is set to False because the final normalization already handles the bias term
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len  * 2,  device = self.args.device) # dim // n_heads is the dimension of each attention head , and max_seq_len * 2 accounts for the maximum sequence length for the rotary embeddings, precompute the complex positional frequencies for the rotary embeddings

    def forward(self,tokens : torch.Tensor , start_ops: int):
        # (B,seqlen)
        batch_size, seq_len = tokens.size() # Get the batch size and sequence length from the input tokens

        # B, seqlen -> B,seqlen, dim 
        h = self.tok_embeddings(tokens) # Convert token indices to embeddings

        # retrieve the pairs (m ,theta ) corresponding to the positions [start_pos,start_pos + seq_len)
        freqs_complex = self.freqs_complex[start_ops:start_ops + seq_len]

        # consecutively pass the embeddings through each encoder block
        for layer in self.layers:
            h = layer(h, start_ops=start_ops, freqs_complex=freqs_complex)

        h = self.norm(h) # Apply final normalization
        logits = self.output_proj(h) # Project to vocabulary size for output logits
        return logits

    def generate(self, tokens: torch.Tensor, max_new_tokens: int):
        """
        Generate new tokens given an initial sequence of tokens.

        Args:
            tokens (torch.Tensor): The input token sequence of shape (B, seqlen).
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            torch.Tensor: The generated token sequence of shape (B, seqlen + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            start_ops = tokens.size(1)
            logits = self.forward(tokens, start_ops=start_ops)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    