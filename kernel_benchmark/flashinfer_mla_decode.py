# Adapt from https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py
import argparse
import os
import sys

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from flashinfer import BatchMLAPagedAttentionWrapper

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from config.model_config import ModelConfig  # noqa E402
from flops.flops import get_mla_absorb_gflops  # noqa E402


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    return t, m


def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean * 1e6


def decode_attention_flashinfer():
    flashinfer_workspace_size = os.environ.get(
        "FLASHINFER_WORKSPACE_SIZE", 384 * 1024 * 1024
    )
    workspace_buffer = torch.empty(
        flashinfer_workspace_size, dtype=torch.int8, device="cuda"
    )

    flashinfer_decode_wrapper = BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend="auto"
    )

    class FlashinferAttention(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q_nope,
            q_pe,
            ckv,
            kpe,
            batch_size,
            kv_len,
            num_local_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            warmup=10,
        ):
            q_indptr = torch.arange(0, batch_size + 1).to(0).int()

            total_tokens = batch_size * kv_len
            kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32).to(0)
            kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
            kv_indices = torch.arange(0, total_tokens).to(0).int()
            page_size = 1

            sm_scale = 1.0 / (
                (128 + qk_rope_head_dim) ** 0.5
            )  # use head dimension before matrix absorption

            # flashinfer_decode_wrapper.end_forward()
            flashinfer_decode_wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                kv_lens,
                num_local_heads,
                kv_lora_rank,
                qk_rope_head_dim,
                page_size,
                False,
                sm_scale,
                q_nope.dtype,
                ckv.dtype,
            )

            for _ in range(warmup):
                o = flashinfer_decode_wrapper.run(
                    q_nope, q_pe, ckv, kpe, return_lse=False
                )

            f = time_fwd(
                flashinfer_decode_wrapper.run,
                q_nope,
                q_pe,
                ckv,
                kpe,
                return_lse=False,
            )

            return f, o

    return FlashinferAttention


def main(args):
    config = ModelConfig(args.config_path)
    fp16_tflops = 148
    num_heads = config.num_attention_heads
    kv_lora_rank = config.kv_lora_rank
    # qk_nope_head_dim = config.qk_nope_head_dim
    qk_rope_head_dim = config.qk_rope_head_dim

    dtype = torch.bfloat16
    kv_cache_dtype = dtype
    if args.kv_cache_dtype == "bf16":
        kv_cache_dtype = torch.bfloat16
    elif args.kv_cache_dtype == "fp8":
        kv_cache_dtype = torch.float8_e4m3fn

    batch_kv_mapping = {
        1: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        16: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        32: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        64: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        128: [1024, 4096, 8192, 16384, 32768, 65536],
        256: [1024, 4096, 8192, 16384],
        512: [1024, 4096, 8192],
    }
    configs = []
    results = []
    for batch_size, kv_len_range in batch_kv_mapping.items():
        configs.extend([(batch_size, kv_len) for kv_len in kv_len_range])

    attn_flashinfer = decode_attention_flashinfer().apply
    for batch_size, kv_len in configs:
        q_nope = torch.randn(
            batch_size, num_heads, kv_lora_rank, dtype=dtype, device="cuda"
        ).to(kv_cache_dtype)
        q_pe = torch.randn(
            batch_size, num_heads, qk_rope_head_dim, dtype=dtype, device="cuda"
        ).to(kv_cache_dtype)
        ckv = torch.randn(
            batch_size * kv_len, 1, kv_lora_rank, dtype=dtype, device="cuda"
        ).to(kv_cache_dtype)
        kpe = torch.randn(
            batch_size * kv_len, 1, qk_rope_head_dim, dtype=dtype, device="cuda"
        ).to(kv_cache_dtype)
        attn_core_gflops, other_gflops = get_mla_absorb_gflops(config, 1, kv_len)
        attn_core_gflops = attn_core_gflops * batch_size

        us_flashinfer, _ = attn_flashinfer(
            q_nope,
            q_pe,
            ckv,
            kpe,
            batch_size,
            kv_len,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        mfu = attn_core_gflops * 1e3 / (fp16_tflops * us_flashinfer)
        print(
            "MLA",
            "  ",
            num_heads,
            "  ",
            kv_lora_rank,
            "  ",
            batch_size,
            "  ",
            kv_len,
            "  ",
            us_flashinfer,
            "  ",
            mfu,
        )

        results.append(
            {
                "dtype": "bf16",
                "kv_dtype": args.kv_cache_dtype,
                "batch_size": batch_size,
                "kv_len": kv_len,
                "latency_us": round(us_flashinfer, 3),
                "mfu": round(mfu, 3),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("attention_benchmark.csv", index=False)


if __name__ == "__main__":
    # calculate_diff()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )

    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bf16", "fp8"],
        default="bf16",
        help="dtype of KV Cache,choices: bf16, fp8",
        required=False,
    )

    args = parser.parse_args()
    main(args)
