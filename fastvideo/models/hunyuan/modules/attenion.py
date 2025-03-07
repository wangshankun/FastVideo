import torch
import torch.nn.functional as F
from einops import rearrange
import inspect

from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward

try:
    from st_attn import sliding_tile_attention
except ImportError:
    print("Could not load Sliding Tile Attention.")
    sliding_tile_attention = None

from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad, flash_attn_sta
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info
import os

def printf(*args, **kwargs):
    """
    An enhanced print function that includes the file name and line number.

    Args:
        *args: The values to be printed.
        **kwargs: The keyword arguments to be passed to the built-in print function.
    """
    # Get the current file name and line number
    frame = inspect.currentframe().f_back
    file_name = os.path.basename(frame.f_code.co_filename)
    line_number = frame.f_lineno

    # Construct the output string
    output = f"{file_name}:{line_number} - "
    output += " ".join(str(arg) for arg in args)

    # Print the output
    print(output, **kwargs)

def attention(
    q,
    k,
    v,
    drop_rate=0,
    attn_mask=None,
    causal=False,
):

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def tile(x, sp_size):
    x = rearrange(x, "b (sp t h w) head d -> b (t sp h w) head d", sp=sp_size, t=30 // sp_size, h=48, w=80)
    return rearrange(x,
                     "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                     n_t=5,
                     n_h=6,
                     n_w=10,
                     ts_t=6,
                     ts_h=8,
                     ts_w=8)


def untile(x, sp_size):
    x = rearrange(x,
                  "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
                  n_t=5,
                  n_h=6,
                  n_w=10,
                  ts_t=6,
                  ts_h=8,
                  ts_w=8)
    return rearrange(x, "b (t sp h w) head d -> b (sp t h w) head d", sp=sp_size, t=30 // sp_size, h=48, w=80)


def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask, mask_strategy=None):

   

    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    text_length = text_mask.sum()

    if get_sequence_parallel_state():
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
        # [b, s, h, d]
   
    hidden_states, encoder_hidden_states = flash_attn_sta(q, k,v, text_mask)
  

    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()
 

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)
    return attn
