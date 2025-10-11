# Adapt from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py
import enum
import random
from typing import Tuple

import deep_gemm
import pandas as pd
import torch
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import (ceil_div, per_block_cast_to_fp8,
                             per_token_cast_to_fp8)


class KernelType(enum.Enum):
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1


def generate_normal(
    m: int,
    n: int,
    k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    accumulate: bool,
    out_dtype: torch.dtype,
    kernel_type: KernelType,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    d = (
        torch.randn((m, n), device="cuda", dtype=out_dtype) * 32
        if accumulate
        else torch.empty((m, n), device="cuda", dtype=out_dtype)
    )
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d

    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (
        per_token_cast_to_fp8(b, use_ue8m0=use_ue8m0)
        if kernel_type.is_1d1d() and accumulate
        else per_block_cast_to_fp8(b, use_ue8m0=use_ue8m0)
    )
    a_fp8 = a_fp8 if major_a.is_k_major() else (a_fp8[0].T.contiguous().T, a_fp8[1])
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].T.contiguous().T, b_fp8[1])
    return a_fp8, b_fp8, c, d, ref_d


def apply_skip_head_mid(d: torch.Tensor, head_splits: Tuple[int, int, int]):
    left, mid, right = head_splits
    m, n = d.shape
    assert n % (left + right) == 0
    num_heads = n // (left + right)

    # Split and insert padding tensor
    d = d.view(m, num_heads, -1)
    d_left = d[:, :, :left]
    d_right = d[:, :, -right:]

    d_mid = torch.zeros((m, num_heads, mid), dtype=d.dtype, device=d.device)
    return torch.cat([d_left, d_mid, d_right], dim=2).view(m, -1)


def test_gemm_skip_head_mid() -> None:
    print("Testing GEMM skip head mid:")
    head_splits = (128, 64, 128)

    major_a, major_b = MajorTypeAB.KMajor, MajorTypeAB.KMajor
    out_dtype, accumulate = torch.bfloat16, False

    for kernel_type in (KernelType.Kernel1D2D,):
        for m in (128, 4096):
            for n, k in [(32768, 512), (8192, 512)]:
                kernel_opt = "1D1D" if kernel_type.is_1d1d() else "1D2D"
                use_ue8m0 = False
                disable_ue8m0_cast = not use_ue8m0

                a, b, _, d, ref_d = generate_normal(
                    m,
                    n,
                    k,
                    major_a,
                    major_b,
                    accumulate,
                    out_dtype,
                    kernel_type,
                    use_ue8m0=use_ue8m0,
                )
                d = apply_skip_head_mid(d, head_splits)
                ref_d = apply_skip_head_mid(ref_d, head_splits)

                deep_gemm.fp8_gemm_nt_skip_head_mid(
                    a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast
                )
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f"{m=}, {n=}, {k=}, {kernel_opt}, {diff:.5f}"

                t = bench_kineto(
                    lambda: deep_gemm.fp8_gemm_nt_skip_head_mid(
                        a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast
                    ),
                    "fp8_gemm",
                    suppress_kineto_output=True,
                )
                print(
                    f" > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}): "
                    f"{t * 1e6:4.0f} us | "
                    f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
                    f"{(count_bytes(a, b, d)) / 1e9 / t:4.0f} GB/s"
                )
    print()


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def generate_cp_test_data(seq_len, seq_len_kv):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    # Select an arbitrary CP rank
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(ceil_div(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def test_paged_mqa_logits():
    print("Testing FP8 Paged MQA Logits:")
    max_model_len = 128 * 1024
    results = []
    gpu_tflops = 296
    for batch_size, next_n in [(64, 1), (64, 2)]:
        for heads, index_dim in [(64, 128)]:
            for avg_kv in (1024, 2048, 4096, 8192, 16384, 32768, 64 * 1024, 128 * 1024):
                num_blocks, blocksize = max_model_len * 3, 64

                q = torch.randn(
                    (batch_size, next_n, heads, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv_cache = torch.randn(
                    (num_blocks, blocksize, 1, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                weights = torch.randn(
                    (batch_size * next_n, heads), device="cuda", dtype=torch.float32
                )

                context_lens = (
                    torch.randint(int(1 * avg_kv), int(1 * avg_kv) + 1, (batch_size,))
                    .cuda()
                    .to(torch.int32)
                )
                max_block_len = (
                    (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                )
                block_tables = torch.zeros(
                    (batch_size, max_block_len), device="cuda", dtype=torch.int32
                )

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = context_lens[i].item()
                    for j in range(ceil_div(ctx_len, blocksize)):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    context_lens, blocksize, deep_gemm.get_num_sms()
                )
                logits = deep_gemm.fp8_paged_mqa_logits(
                    q_fp8,
                    kv_cache_fp8,
                    weights,
                    context_lens,
                    block_tables,
                    schedule_metadata,
                    max_model_len,
                    clean_logits=True,
                )

                ref_logits = ref_fp8_paged_mqa_logits(
                    q, kv_cache, weights, context_lens, block_tables, max_model_len
                )
                positions = (
                    torch.arange(max_model_len, device="cuda")
                    .unsqueeze(0)
                    .expand(batch_size * next_n, -1)
                )
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = (
                    torch.arange(batch_size * next_n, device="cuda") % next_n
                )
                ref_neginf_mask = ~(
                    positions
                    <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)
                )

                neginf_mask = logits == float("-inf")
                assert torch.equal(neginf_mask, ref_neginf_mask)

                logits = logits.masked_fill(neginf_mask, 0)
                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"

                sum_lens = sum(context_lens.to(torch.int64))
                tflops = 2 * sum_lens * next_n * heads * index_dim / 1e12
                input_bytes = (
                    count_bytes(q_fp8, weights, context_lens)
                    + sum_lens * (index_dim + 4)
                    + (sum_lens / blocksize) * 4
                )
                output_bytes = sum_lens * next_n * 4
                t, clean_t = bench_kineto(
                    lambda: deep_gemm.fp8_paged_mqa_logits(
                        q_fp8,
                        kv_cache_fp8,
                        weights,
                        context_lens,
                        block_tables,
                        schedule_metadata,
                        max_model_len,
                        clean_logits=True,
                    ),
                    ("fp8_paged_mqa_logits", "clean_logits"),
                )
                clean_bytes = (
                    batch_size * next_n * max_model_len - neginf_mask.sum().item()
                ) * 4 + count_bytes(context_lens)
                print(
                    f" > BSZ={batch_size:3}, NextN={next_n:1}, H={heads:2}, D={index_dim:2}, L={avg_kv:6}: "
                    f"{tflops / t:4.0f} TFLOPS, {t * 1e6:3.0f} us, "
                    f"{(input_bytes + output_bytes) / t / 1e9:4.0f} GB/s | "
                    f"clean: {clean_t * 1e6:3.0f} us, {clean_bytes / clean_t / 1e9:4.0f} GB/s"
                )
                results.append(
                    {
                        "batchsize": batch_size,
                        "next_n": next_n,
                        "s_kv": avg_kv,
                        "latency_us": round(t * 1e6, 3),
                        "mfu": round(tflops.item() / t / gpu_tflops, 3),
                    }
                )
    print()
    df = pd.DataFrame(results)
    df.to_csv("paged_mqa_logits.csv", index=False)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    test_gemm_skip_head_mid()

    test_paged_mqa_logits()
