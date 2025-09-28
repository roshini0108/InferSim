# Adapt from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_fp8.py
import argparse
import enum
import os
import random
import sys
import time
from typing import Generator

import deep_gemm
import pandas as pd
import torch
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from config.model_config import ModelConfig  # noqa E402


class KernelType(enum.Enum):
    # For SM100 GEMMs
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


def enumerate_normal(use_bf16: bool = False) -> Generator:
    for kernel_type in (KernelType.Kernel1D2D,):
        for major_a, major_b in ((MajorTypeAB.KMajor, MajorTypeAB.KMajor),):
            for out_dtype in (torch.bfloat16,):
                for accumulate in (
                    (False,)
                    if out_dtype == torch.bfloat16 or kernel_type.is_1d2d()
                    else (False, True)
                ):
                    yield kernel_type, major_a, major_b, accumulate, out_dtype


def generate_normal(
    m: int,
    n: int,
    k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    accumulate: bool,
    out_dtype: torch.dtype,
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

    a_fp8, b_fp8 = (
        per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0),
        per_block_cast_to_fp8(b, use_ue8m0=use_ue8m0),
    )
    a_fp8 = a_fp8 if major_a.is_k_major() else (a_fp8[0].T.contiguous().T, a_fp8[1])
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].T.contiguous().T, b_fp8[1])
    return a_fp8, b_fp8, c, d, ref_d


def test_gemm(m, k, n) -> None:
    print("Testing GEMM:")
    for kernel_type, major_a, major_b, accumulate, out_dtype in enumerate_normal():
        major_opt = "N" if major_a.is_k_major() else "T"
        major_opt += "T" if major_b.is_k_major() else "N"
        out_opt = "BF16"
        acc_opt = f"acc={int(accumulate)}"
        kernel_opt = "1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = False
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(
                m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0
            )
            func_name = f"fp8_gemm_{major_opt.lower() if test_alias else 'nt'}"
            if test_alias:
                a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(
                a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast
            )
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, (
                f"{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, "
                f"{diff:.5f}, alias={test_alias}"
            )
        a, b, c, d, ref_d = generate_normal(
            m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0
        )

        # Test launch overhead
        launch_start_t = time.time_ns()
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
        launch_end_t = time.time_ns()
        torch.cuda.synchronize()

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        print(
            f" > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): "
            f"launch {(launch_end_t - launch_start_t) / 1e3:4.0f} us | {t * 1e6:4.0f} us | "
            f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s"
        )
        tflops = 2 * m * n * k / t / 1e12
        return t * 1e6, tflops


def main(args) -> None:
    print("Testing grouped masked GEMM:")
    results = []

    for m in [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        4096,
        8192,
        16384,
        32768,
        64 * 1024,
        128 * 1024,
    ]:
        t, tflops = test_gemm(m, args.k, args.n)
        results.append(
            {
                "m": m,
                "k": args.k,
                "n": args.n,
                "latency_us": round(t, 3),
                "mfu": round(tflops / args.gpu_tflops, 3),
            }
        )
    print()
    df = pd.DataFrame(results)
    df.to_csv("gemm.csv", index=False)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print("Library path:")
    print(f" > {deep_gemm.__path__}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=1024, help="[m, k] * [k, n]")
    parser.add_argument("-n", type=int, default=1024, help="[m, k] * [k, n]")
    parser.add_argument("--gpu-tflops", type=int, default=296, help="GPU FP8 TFLOPS")
    args = parser.parse_args()
    main(args)
