"""
TriMul (outgoing) forward pass with a fused Triton kernel.

Algorithm
---------
1. Layer‑norm the input tensor `x` over the channel dimension `dim`.
2. Project the normalized tensor to a hidden dimension `hidden_dim` twice:
   * left  = Linear(x, left_proj_weight)
   * right = Linear(x, right_proj_weight)
3. (Optional) Apply the pairwise mask.
4. Compute three sigmoid gates (left, right, out) from another linear projection.
5. Modulate `left` and `right` with their respective gates.
6. Core contraction:
   out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
   This is a batched matrix multiplication per hidden channel:
   out = left @ rightᵀ  (shape (B, hidden_dim, N, N)).
   The contraction is performed by a custom Triton kernel that tiles the
   (i, j) output plane and accumulates along the k‑dimension.
7. Apply a second layer‑norm over the hidden dimension, multiply by the
   output gate and finally linearly project back to `dim`.
All operations are carried out in float32; the Triton kernel works on
float32 tensors and uses block‑wise loading, masking and `tl.dot` for the
inner‑product accumulation.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Triton kernel that computes the triangle‑multiplicative contraction.
# --------------------------------------------------------------------------- #
@triton.jit
def tri_mul_outgoing_kernel(
    left_ptr,               # const float*  [B, N, N, D]
    right_ptr,              # const float*  [B, N, N, D]
    out_ptr,                # float*        [B, N, N, D]

    batch, N, hidden,       # scalar sizes
    # strides of left
    stride_left_batch, stride_left_i, stride_left_k, stride_left_d,
    # strides of right
    stride_right_batch, stride_right_i, stride_right_k, stride_right_d,
    # strides of out
    stride_out_batch, stride_out_i, stride_out_j, stride_out_d,

    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute
        out[b, i, j, d] = Σ_k left[b, i, k, d] * right[b, j, k, d]
    using a tiled matmul (left @ rightᵀ) for each (b, d) pair.
    """
    pid_m = tl.program_id(0)          # tile over i
    pid_n = tl.program_id(1)          # tile over j
    pid_bd = tl.program_id(2)         # combined batch * hidden index

    batch_idx = pid_bd // hidden      # actual batch index
    d_idx     = pid_bd % hidden       # hidden‑channel index

    # ----------------------  i / j coordinates --------------------------- #
    i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    i_mask = i < N
    j_mask = j < N

    # ------------------------------------------------------------------- #
    # Accumulator for the tile (M × N)
    # ------------------------------------------------------------------- #
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension in tiles
    num_k_tiles = tl.cdiv(N, BLOCK_K)
    for k_tile in range(num_k_tiles):
        k_offset = k_tile * BLOCK_K
        k = k_offset + tl.arange(0, BLOCK_K)
        k_mask = k < N

        # left tile: shape (BLOCK_M, BLOCK_K)
        left_ptrs = (
            left_ptr
            + batch_idx * stride_left_batch
            + d_idx    * stride_left_d
            + i[:, None] * stride_left_i
            + k[None, :] * stride_left_k
        )
        left_tile = tl.load(
            left_ptrs,
            mask=i_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # right tile (transposed): shape (BLOCK_K, BLOCK_N)
        # we need right[b, j, k, d] → treat k as row, j as column
        right_ptrs = (
            right_ptr
            + batch_idx * stride_right_batch
            + d_idx    * stride_right_d
            + j[None, :] * stride_right_i
            + k[:, None] * stride_right_k
        )
        right_tile = tl.load(
            right_ptrs,
            mask=k_mask[:, None] & j_mask[None, :],
            other=0.0,
        )

        # accumulate the matrix product of the two tiles
        acc += tl.dot(left_tile, right_tile)   # (M,K)·(K,N) → (M,N)

    # --------------------------- store ----------------------------------- #
    out_ptrs = (
        out_ptr
        + batch_idx * stride_out_batch
        + d_idx    * stride_out_d
        + i[:, None] * stride_out_i
        + j[None, :] * stride_out_j
    )
    tl.store(out_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])


# --------------------------------------------------------------------------- #
# Entry point expected by the autograder
# --------------------------------------------------------------------------- #
def custom_kernel(data):
    """
    Forward pass of the TriMul (outgoing) module using a Triton kernel for
    the O(N³·D) contraction.

    Args:
        data: tuple (input_tensor, mask_tensor, weights_dict, config_dict)
            - input_tensor : [B, N, N, dim]  torch.Tensor
            - mask_tensor  : [B, N, N]       torch.Tensor (binary or float)
            - weights_dict : mapping of weight names to torch.Tensors
            - config_dict  : must contain "dim" and "hidden_dim"

    Returns:
        torch.Tensor of shape [B, N, N, dim] (float32)
    """
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    # work in float32 for numerical stability
    x = input_tensor.to(torch.float32)

    dim = config["dim"]
    hidden = config["hidden_dim"]
    B, N, _, _ = x.shape

    # --------------------------------------------------------------------- #
    # 1. First layer‑norm
    # --------------------------------------------------------------------- #
    x = F.layer_norm(
        x,
        normalized_shape=(dim,),
        weight=weights["norm.weight"].to(device).to(torch.float32),
        bias=weights["norm.bias"].to(device).to(torch.float32),
    )

    # --------------------------------------------------------------------- #
    # 2. Linear projections (no bias)
    # --------------------------------------------------------------------- #
    left = F.linear(x, weights["left_proj.weight"].to(device).to(torch.float32))
    right = F.linear(x, weights["right_proj.weight"].to(device).to(torch.float32))

    # --------------------------------------------------------------------- #
    # 3. Optional mask
    # --------------------------------------------------------------------- #
    if mask is not None:
        mask_exp = mask.to(torch.float32).unsqueeze(-1)   # [B,N,N,1]
        left = left * mask_exp
        right = right * mask_exp

    # --------------------------------------------------------------------- #
    # 4. Gates (sigmoid after linear)
    # --------------------------------------------------------------------- #
    left_gate = torch.sigmoid(
        F.linear(x, weights["left_gate.weight"].to(device).to(torch.float32))
    )
    right_gate = torch.sigmoid(
        F.linear(x, weights["right_gate.weight"].to(device).to(torch.float32))
    )
    out_gate = torch.sigmoid(
        F.linear(x, weights["out_gate.weight"].to(device).to(torch.float32))
    )

    left = left * left_gate
    right = right * right_gate

    # --------------------------------------------------------------------- #
    # 5. Triangular contraction via Triton
    # --------------------------------------------------------------------- #
    out = torch.empty_like(left)   # shape [B,N,N,hidden]

    # Block sizes – tuned for typical N ≤ 1024 on an A200
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(N, BLOCK_M),          # i‑tiles
        triton.cdiv(N, BLOCK_N),          # j‑tiles
        B * hidden,                       # combined (batch, hidden) dimension
    )
    tri_mul_outgoing_kernel[grid](
        left,
        right,
        out,
        B,
        N,
        hidden,
        left.stride(0), left.stride(1), left.stride(2), left.stride(3),
        right.stride(0), right.stride(1), right.stride(2), right.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # --------------------------------------------------------------------- #
    # 6. Second layer‑norm, output gate, final linear projection
    # --------------------------------------------------------------------- #
    out = F.layer_norm(
        out,
        normalized_shape=(hidden,),
        weight=weights["to_out_norm.weight"].to(device).to(torch.float32),
        bias=weights["to_out_norm.bias"].to(device).to(torch.float32),
    )
    out = out * out_gate
    out = F.linear(out, weights["to_out.weight"].to(device).to(torch.float32))

    # Ensure final dtype is float32 as required
    return out.to(torch.float32)