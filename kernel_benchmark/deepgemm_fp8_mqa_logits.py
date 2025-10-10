import enum
import random
from typing import Tuple

import deep_gemm
import pandas as pd
import torch
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import (per_block_cast_to_fp8,
                             per_custom_dims_cast_to_fp8,
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


def ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    cost_only: bool = False,
):
    seq_len_kv = kv.shape[0]

    if cost_only:
        start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
        end = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
        count_ones_per_row = (end - start).clamp(min=0)
        return count_ones_per_row.sum()

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    cost = mask.sum()
    return logits, cost


def test_mqa_logits():
    print("Testing FP8 MQA Logits:")
    results = []
    gpu_tflops = 296
    num_heads, head_dim = 64, 128
    for seq_len in (2048, 4096):
        for seq_len_kv in (4096, 8192, 16384, 32768, 65536, 131072):
            for disable_cp in (True,):
                q = torch.randn(
                    seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
                )
                kv = torch.randn(
                    seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16
                )
                weights = torch.randn(
                    seq_len, num_heads, device="cuda", dtype=torch.float32
                )

                if disable_cp:
                    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
                    ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
                        seq_len_kv - seq_len
                    )
                else:
                    ks, ke = generate_cp_test_data(seq_len, seq_len_kv)

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
                logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

                do_check = seq_len_kv < 32768
                if do_check:
                    ref_logits, ref_cost = ref_fp8_mqa_logits(
                        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
                    )

                    ref_neginf_mask = ref_logits == float("-inf")
                    neginf_mask = logits == float("-inf")
                    assert torch.equal(neginf_mask, ref_neginf_mask)

                    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                    logits = logits.masked_fill(neginf_mask, 0)
                    diff = calc_diff(logits, ref_logits)
                    assert diff < 1e-3, f"{diff=}"
                else:
                    ref_cost = ref_fp8_mqa_logits(
                        q=q,
                        kv=kv,
                        weights=weights,
                        cu_seqlen_ks=ks,
                        cu_seqlen_ke=ke,
                        cost_only=True,
                    )

                tflops = 2 * ref_cost * num_heads * head_dim / 1e12
                t, clean_t = bench_kineto(
                    lambda: deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke),
                    ("fp8_mqa_logits", "clean_logits"),
                )
                clean_bytes = (seq_len * seq_len_kv - ref_cost) * 4 + count_bytes(
                    ks, ke
                )
                print(
                    f" > S={seq_len:4}, SKV={seq_len_kv:6}, H={num_heads:3}, D={head_dim:3}, CP={0 if disable_cp else 1}: "
                    f"{tflops / t:4.0f} TFLOPS, {t * 1e6:4.0f} us, "
                    f"{(count_bytes(q_fp8, kv_fp8, weights, ks, ke) + ref_cost * 4) / t / 1e9:4.0f} GB/s | "
                    f"clean: {clean_t * 1e6:3.0f} us, {clean_bytes / clean_t / 1e9:4.0f} GB/s"
                )
                results.append(
                    {
                        "s_q": seq_len,
                        "s_kv": seq_len_kv,
                        "latency_us": round(t * 1e6, 3),
                        "mfu": round(tflops.item() / t / gpu_tflops, 3),
                    }
                )
    print()
    df = pd.DataFrame(results)
    df.to_csv("mqa_logits.csv", index=False)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    test_gemm_skip_head_mid()

    test_mqa_logits()
