from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
import torch

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


def flash_attn_sta_no_pad(qkv, key_padding_mask, img_len, txt_len, causal=False, dropout_p=0.0, softmax_scale=None):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]

    x = rearrange(qkv, "b s three h d -> b s (three h d)")

    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    
    '''
    output_unpad, lse, _ = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    #print(x_unpad.shape, cu_seqlens, max_s, output_unpad.shape)
    '''

    img_len = img_len
    txt_len = txt_len
    cu_img_len =  torch.tensor([0, img_len], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    cu_txt_len = torch.tensor([0, txt_len], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    x_unpad_img, x_unpad_txt = torch.split(x_unpad, [img_len, txt_len])
    img_q, img_kv = torch.split(x_unpad_img, [1, 2], dim=1)
    txt_q, txt_kv = torch.split(x_unpad_txt, [1, 2], dim=1)

    img_q = img_q.squeeze(1)
    txt_q = txt_q.squeeze(1)     

    X_img2img, lse_img2img,  _ = flash_attn_varlen_kvpacked_func(
        img_q,
        img_kv,
        cu_seqlens_q = cu_img_len,
        max_seqlen_q = img_len,
        cu_seqlens_k = cu_img_len,
        max_seqlen_k = img_len,
        dropout_p = dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
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
    output_unpad = torch.cat([merged_img_out.to(img_q.dtype), merged_text_out.to(txt_q.dtype)], dim=0)


    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )

    return output