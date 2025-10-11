import argparse
import dataclasses
import math
import random
from typing import Optional, Tuple

import flash_mla
import pandas as pd
import torch
import triton


def cdiv(x: int, y: int):
    return (x + y - 1) // y


def quantize_k_cache(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    dv: int,
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Return a tensor with shape (num_blocks, block_size, h_k, dv + 4(dv/tile_size) + t(d-dv)) of dtype uint8_t, where t = input_k_cache.element_size()
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py or README.md
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    result = torch.empty(
        (num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)),
        dtype=torch.float8_e4m3fn,
        device=input_k_cache.device,
    )
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv : dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4 :].view(input_k_cache.dtype)
    result_k_rope_part[:] = input_k_cache[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_factors_inv = (
            torch.abs(
                input_k_cache[..., tile_idx * tile_size : (tile_idx + 1) * tile_size]
            )
            .max(dim=-1)
            .values
            / 448.0
        )  # [num_blocks, block_size]
        result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

        cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
        cur_quantized_nope = (
            input_k_cache[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].float()
            / cur_scale_factors_inv.float()
        ).to(torch.float8_e4m3fn)
        result_k_nope_part[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_quantized_nope
        )

    result = result.view(num_blocks, block_size, 1, -1)
    return result


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv : dv + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[..., dv + num_tiles * 4 :].view(torch.bfloat16)
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        cur_nope = input_nope[
            ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        )

    result = result.view(num_blocks, block_size, 1, d)
    return result


@dataclasses.dataclass
class TestParam:
    b: int  # Batch size
    s_q: int  # Number of queries for one request
    s_k: int  # Seq len, or mean seq len if varlen == True
    is_varlen: bool
    is_causal: bool
    is_fp8: bool
    topk: Optional[int] = None
    test_performance: bool = True
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    block_size: int = 64
    h_q: int = 128  # Number of q heads
    h_kv: int = 1  # Number of kv heads
    d: int = 576  # Q/K head dim (= dv + RoPE dim)
    dv: int = 512  # V head dim
    seed: int = 0


def generate_test_data(
    t: TestParam,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Generate test data from a given configuration
    Return: [cache_seqlens, q, block_table, blocked_k]
    Pay attention: This function changes the random seed
    """
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    torch.backends.cudnn.deterministic = True

    assert t.h_q % t.h_kv == 0

    cache_seqlens_cpu = torch.full((t.b,), t.s_k, dtype=torch.int32, device="cpu")
    if t.is_varlen:
        for i in range(t.b):
            cache_seqlens_cpu[i] = max(random.normalvariate(t.s_k, t.s_k / 2), t.s_q)

    if t.have_zero_seqlen_k:
        zeros_mask = torch.randn(t.b, dtype=torch.float32, device="cpu") > 0
        cache_seqlens_cpu[zeros_mask] = 0

    max_seqlen = cache_seqlens_cpu.max().item()
    max_seqlen_pad = cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.cuda()

    q = torch.randn(t.b, t.s_q, t.h_q, t.d)
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(
        t.b * max_seqlen_pad // t.block_size, dtype=torch.int32
    ).view(t.b, max_seqlen_pad // t.block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(
        t.b, -1
    )
    blocked_k = torch.randn(block_table.numel(), t.block_size, t.h_kv, t.d) / 10
    blocked_k.clamp_(min=-1.0, max=1.0)

    if t.topk is None:
        for i in range(t.b):
            cur_len = cache_seqlens_cpu[i].item()
            cur_num_blocks = cdiv(cur_len, t.block_size)
            blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
            if cur_len % t.block_size != 0:
                blocked_k[block_table[i][cur_num_blocks - 1]][
                    cur_len % t.block_size :
                ] = float("nan")
            block_table[i][cur_num_blocks:] = 2147480000
        return cache_seqlens, q, block_table, blocked_k, None, None
    else:
        block_table_cpu = block_table.cpu()
        abs_indices = torch.empty(t.b, t.s_q, t.topk, dtype=torch.int32, device="cpu")
        indices_in_kvcache = torch.empty(
            t.b, t.s_q, t.topk, dtype=torch.int32, device="cpu"
        )
        for i in range(t.b):
            # Generate indices
            for j in range(t.s_q):
                cur_abs_indices = torch.randperm(
                    int(cache_seqlens_cpu[i].item()), device="cpu"
                )[: t.topk]
                cur_blocked_indices = block_table_cpu[
                    i, cur_abs_indices // t.block_size
                ] * t.block_size + (cur_abs_indices % t.block_size)
                if len(cur_abs_indices) < t.topk:
                    pad_len = t.topk - len(cur_abs_indices)
                    cur_abs_indices = torch.cat(
                        [cur_abs_indices, torch.full((pad_len,), -1, device="cpu")]
                    )
                    cur_blocked_indices = torch.cat(
                        [cur_blocked_indices, torch.full((pad_len,), -1, device="cpu")]
                    )

                # Mask KV
                perm = torch.randperm(t.topk, device="cpu")
                cur_abs_indices = cur_abs_indices[perm]
                cur_blocked_indices = cur_blocked_indices[perm]

                # Fill it with invalid indices if needed
                if t.is_all_indices_invalid:
                    cur_abs_indices.fill_(-1)
                    cur_blocked_indices.fill_(-1)

                abs_indices[i, j, :] = cur_abs_indices
                indices_in_kvcache[i, j, :] = cur_blocked_indices

        # Mask nonused KV as NaN
        all_indices = indices_in_kvcache.flatten().tolist()
        all_indices = list(set(all_indices))
        if -1 in all_indices:
            all_indices.remove(-1)
        all_indices = torch.tensor(all_indices, dtype=torch.int32, device="cpu")

        blocked_k = blocked_k.view(-1, t.h_kv, t.d)
        nonused_indices_mask = torch.ones(
            blocked_k.size(0) * blocked_k.size(1), dtype=torch.bool, device="cpu"
        )
        nonused_indices_mask[all_indices] = False
        blocked_k[nonused_indices_mask, :, :] = float("nan")
        blocked_k = blocked_k.view(-1, t.block_size, t.h_kv, t.d)

        abs_indices = abs_indices.to(q.device)
        indices_in_kvcache = indices_in_kvcache.to(q.device)

        return cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache


def reference_torch(
    cache_seqlens: torch.Tensor,  # [batch_size]
    block_table: torch.Tensor,  # [batch_size, ?]
    q: torch.Tensor,  # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,  # [?, block_size, h_kv, d]
    dv: int,
    is_causal: bool,
    indices: Optional[torch.Tensor] = None,  # [batch_size, s_q, topk]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch
    """

    def get_topk_attn_mask(s_q: int, s_k: int, indices: torch.Tensor):
        mask = torch.zeros(s_q, s_k, dtype=torch.bool)
        for i in range(s_q):
            cur_indices = indices[i]
            valid_indices = cur_indices[cur_indices != -1]
            mask[i, valid_indices] = True
        return mask

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,  # [h_q, s_q, d]
        kv: torch.Tensor,  # [h_kv, s_k, d]
        dv: int,
        is_causal,
        indices: Optional[torch.Tensor],  # [s_q, topk]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_q = query.size(0)
        h_kv = kv.size(0)
        s_q = query.shape[-2]
        s_k = kv.shape[-2]
        query = query.float()
        kv = kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0
        attn_weight = query @ kv.transpose(-2, -1)  # [h_q, s_q, s_k]
        if (is_causal and query.size(1) > 1) or indices is not None:
            mask = torch.ones(s_q, s_k, dtype=torch.bool)
            if is_causal:
                assert indices is None
                mask = mask.tril(diagonal=s_k - s_q)
            if indices is not None:
                mask &= get_topk_attn_mask(s_q, s_k, indices)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)  # [h_q, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]  # [h_q, s_q, dv]
        # Correct for q tokens which has no attendable k
        lonely_q_mask = lse == float("-inf")
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output, lse

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32)
    for i in range(b):
        cur_len = cache_seqlens_cpu[i].item()
        cur_num_blocks = cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0:cur_num_blocks]
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]
        cur_out, cur_lse = scaled_dot_product_attention(
            i,
            q[i].transpose(0, 1),
            cur_kv.transpose(0, 1),
            dv,
            is_causal,
            indices[i] if indices is not None else None,
        )
        out_ref[i] = cur_out.transpose(0, 1)
        lse_ref[i] = cur_lse
    out_ref = out_ref.to(torch.bfloat16)
    return out_ref, lse_ref


@torch.inference_mode()
def test_flash_mla(t: TestParam, results):
    print("-------------------------------")
    print(f"Running on {t}...")
    gpu_tflops = 148

    # Generating test data
    torch.cuda.synchronize()
    cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache = (
        generate_test_data(t)
    )

    if t.is_fp8:
        # The quantization error may be too large to be distinguished from wrong kernels
        # So we quantize and de-quantize kv-cache here to mitigate quantization error
        blocked_k_quantized = quantize_k_cache(blocked_k, t.dv, 128)
        blocked_k_dequantized = dequantize_k_cache(blocked_k_quantized)
        blocked_k = blocked_k_dequantized

    # Get schedule metadata
    torch.cuda.synchronize()
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        cache_seqlens, t.s_q * t.h_q // t.h_kv, t.h_kv, t.h_q, t.is_fp8, t.topk
    )
    torch.cuda.synchronize()

    def run_flash_mla():
        return flash_mla.flash_mla_with_kvcache(
            q,
            blocked_k if not t.is_fp8 else blocked_k_quantized,  # type: ignore
            block_table,
            cache_seqlens,
            t.dv,
            tile_scheduler_metadata,
            num_splits,
            causal=t.is_causal,
            is_fp8_kvcache=t.is_fp8,
            indices=indices_in_kvcache,
        )

    out_ans, lse_ans = run_flash_mla()
    out_ref, lse_ref = reference_torch(
        cache_seqlens, block_table, q, blocked_k, t.dv, t.is_causal, abs_indices
    )
    # assert check_is_allclose("out", out_ans, out_ref, abs_tol=8e-4, rel_tol=2.01/128, cos_diff_tol=5e-6)
    # assert check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)

    if t.test_performance:
        time_usage: float = triton.testing.do_bench(run_flash_mla) / 1000  # type: ignore
        mean_attended_seqlens = (
            cache_seqlens.float().mean().item() if t.topk is None else t.topk
        )
        compute_volume_flop = (
            t.b
            * t.h_q
            * t.s_q
            * sum(
                [
                    2 * t.d * mean_attended_seqlens,  # Q * K^T
                    2 * mean_attended_seqlens * t.dv,  # attention * V
                ]
            )
        )
        q_elem_size = torch.bfloat16.itemsize
        kv_token_size = 656 if t.is_fp8 else t.d * torch.bfloat16.itemsize
        memory_volume_B = t.b * sum(
            [
                t.s_q * t.h_q * (t.d * q_elem_size),  # Q
                (t.s_q if t.topk is not None else 1)
                * mean_attended_seqlens
                * t.h_kv
                * kv_token_size,  # K/V
                t.s_q * t.h_q * (t.dv * q_elem_size),  # Output
            ]
        )
        achieved_tflops = compute_volume_flop / time_usage / 1e12
        achieved_gBps = memory_volume_B / time_usage / 1e9

        print(
            f"{time_usage * 1000:.3f} ms, {achieved_tflops:.0f} TFLOPS, {achieved_gBps:.0f} GB/s"
        )
        results.append(
            {
                "dtype": "bf16",
                "kv_dtype": "fp8" if t.is_fp8 else "bf16",
                "batch_size": t.b,
                "kv_len": t.s_k,
                "latency_us": round(time_usage * 1e6, 3),
                "mfu": round(achieved_tflops / gpu_tflops, 3),
            }
        )


def main(torch_dtype):
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    correctness_cases = [
        TestParam(
            b, s_q, s_k, is_varlen, is_causal, is_fp8, topk, test_performance=False
        )
        for b in [1, 2, 6, 64]
        for s_q in [1, 2, 4]
        for s_k in [20, 140, 4096]
        for is_varlen in [False, True]
        for is_causal in [False, True]
        for (is_fp8, topk) in [(False, None), (True, 128), (True, 2048)]
        if not (is_causal and topk is not None)
    ]

    corner_cases = [
        # Cases where all topk indices are invalid
        TestParam(
            128,
            2,
            4096,
            is_varlen=True,
            is_causal=False,
            is_fp8=True,
            topk=topk,
            test_performance=False,
            is_all_indices_invalid=True,
        )
        for topk in [128, 2048, 4096]
    ] + [
        # Cases where some kv cache have zero length
        TestParam(
            128,
            2,
            4096,
            is_varlen=True,
            is_causal=is_causal,
            is_fp8=is_fp8,
            topk=topk,
            test_performance=False,
            have_zero_seqlen_k=True,
        )
        for (is_causal, is_fp8, topk) in [
            (False, False, None),
            (True, False, None),
            (False, True, 128),
            (False, True, 2048),
        ]
    ]

    results = []
    performance_cases = [
        TestParam(
            64,
            s_q,
            s_k,
            is_varlen=True,
            is_causal=is_causal,
            is_fp8=is_fp8,
            topk=topk,
            test_performance=True,
        )
        for (is_causal, is_fp8, topk) in [
            # (False, False, None),
            # (True, False, None),
            (False, True, 2048),
        ]
        for s_q in [1]
        for s_k in [1024, 2048, 4096, 8192, 16384, 32768, 64 * 1024, 128 * 1024]
    ]

    testcases = correctness_cases + corner_cases + performance_cases

    # Prune out unsupported cases
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if cc_major == 10:
        testcases = [t for t in testcases if (t.is_fp8 and t.topk is not None)]

    for testcase in testcases:
        test_flash_mla(testcase, results)

    df = pd.DataFrame(results)
    df.to_csv("dsa_decode.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type to use for testing (bf16 or fp16)",
    )

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16

    main(torch_dtype)
