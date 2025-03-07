from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
import torch
import torch.nn.functional as F
from fastvideo.models.flex_sta_ref import get_sliding_tile_attention_mask

from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention, dynamic=False)

def merge_attention_blocks(blocks):
    out = None
    lse = None
    for block_out, block_lse in blocks:
        # 确保块输出和lse为float32以保证数值稳定性
        block_out = block_out.to(torch.float32)
        # 调整lse形状为 (h, seq_len, 1)
        block_lse = block_lse.transpose(0, 1).unsqueeze(-1)
        
        if out is None:
            out = block_out
            lse = block_lse
        else:
            # 计算新的logsumexp
            new_lse = torch.logaddexp(lse, block_lse)
            # 重整化并合并输出
            out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
            lse = new_lse
    return out, lse

def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output

def flash_attn_sta(q, k, v, text_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    img_q, txt_q = q
    img_k, txt_k = k
    img_v, txt_v = v
    
    txt_max_len = txt_q.size(1)
    img_len     = img_q.size(1)
    txt_len     = text_mask.sum().item()
    cu_img_len  = torch.tensor([0, img_len], dtype=torch.int32, device=img_q.device)
    cu_txt_len  = torch.tensor([0, txt_len], dtype=torch.int32, device=img_q.device)
    
    #取有效长度txt
    txt_q = txt_q[:, :txt_len, :, :]
    txt_k = txt_k[:, :txt_len, :, :]
    txt_v = txt_v[:, :txt_len, :, :]

    txt_kv = torch.stack([txt_k, txt_v], dim=2)
    img_kv = torch.stack([img_k, img_v], dim=2)
    #flex_attention格式要求 b h s d
    img_q = rearrange(img_q, 'b s h d -> b h s d')
    img_k = rearrange(img_k, 'b s h d -> b h s d')
    img_v = rearrange(img_v, 'b s h d -> b h s d')

    img_q = img_q.contiguous()
    img_k = img_k.contiguous()
    img_v = img_v.contiguous()

    X_img2img_   =  torch.empty_like(img_q)
    lse_img2img_ =  torch.empty(img_q.shape[:3], dtype=img_q.dtype, device=img_q.device)

    kernel_size = (5, 6, 10)
    mask = get_sliding_tile_attention_mask(kernel_size, (6, 8, 8), (30, 48, 80), 0, 'cuda', 0)

    X_img2img_, lse_img2img_ = flex_attention(img_q, img_k, img_v, block_mask=mask, return_lse=True)
    X_img2img = rearrange(X_img2img_, 'b h s d -> b s h d')
    X_img2img = X_img2img.squeeze(0)
    lse_img2img = lse_img2img_.squeeze(0)

    #flash attention格式 s h d， 不支持batch
    img_q = rearrange(img_q, 'b h s d -> b s h d')
    img_q  = img_q.squeeze(0)
    img_kv = img_kv.squeeze(0)
    txt_q  = txt_q.squeeze(0)
    txt_kv = txt_kv.squeeze(0)
    X_img2text, lse_img2text, _ = flash_attn_varlen_kvpacked_func(
                img_q,
                txt_kv,
                cu_seqlens_q = cu_img_len,
                max_seqlen_q = img_len,
                cu_seqlens_k = cu_txt_len,
                max_seqlen_k = txt_len,
                dropout_p = dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                return_attn_probs=True,
    )

    X_text2img, lse_text2img, _   = flash_attn_varlen_kvpacked_func(
                txt_q,
                img_kv,
                cu_seqlens_q = cu_txt_len,
                max_seqlen_q = txt_len,
                cu_seqlens_k = cu_img_len,
                max_seqlen_k = img_len,
                dropout_p = dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                return_attn_probs=True,
    )
    X_text2text, lse_text2text, _ = flash_attn_varlen_kvpacked_func(
                txt_q,
                txt_kv,
                cu_seqlens_q = cu_txt_len,
                max_seqlen_q = txt_len,
                cu_seqlens_k = cu_txt_len,
                max_seqlen_k = txt_len,
                dropout_p = dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                return_attn_probs=True,
    )

    # 合并图像查询部分
    merged_img_out, _ = merge_attention_blocks([
    (X_img2img, lse_img2img),
    (X_img2text, lse_img2text)
    ])

    # 合并文本查询部分
    merged_text_out, _ = merge_attention_blocks([
    (X_text2img, lse_text2img),
    (X_text2text, lse_text2text)
    ])

    hidden_states = merged_img_out.to(txt_q.dtype)
    encoder_hidden_states = merged_text_out.to(txt_q.dtype)
    encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, 0, 0, txt_max_len - txt_len))
    #加上batch维度
    hidden_states = hidden_states.unsqueeze(0)
    encoder_hidden_states = encoder_hidden_states.unsqueeze(0)

    return hidden_states, encoder_hidden_states