import argparse
import math

from comm.comm import Comm
from config.model_config import ModelConfig
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size
from layers.attn import create_attention
from layers.moe import MoE
from params.params import get_attn_params_size, get_expert_params_size


def prefill(args, config, gpu, kvcache_bytes):
    print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
    print("{:<40} {:<10}".format("Max prefill tokens:", args.max_prefill_tokens))
    attn = create_attention(config, args.use_fp8_gemm, args.use_fp8_kv)
    attn_core_time = attn.prefill_attn_core(
        args.target_isl, kvcache_bytes, args.device_type
    )
    attn_other_time = attn.prefill_attn_others(
        args.max_prefill_tokens, args.device_type
    )
    attn_core_time *= math.ceil(args.max_prefill_tokens / args.target_isl)

    moe = MoE(config, args.use_fp8_gemm)
    moe_time = moe.prefill_moe(
        args.max_prefill_tokens, args.device_type, args.world_size
    )

    comm = Comm(config, gpu, args.world_size, args.num_nodes, args.enable_deepep)
    comm_time1, comm_time2 = comm.prefill_comm(args.max_prefill_tokens)
    print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
    print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

    num_tokens = args.max_prefill_tokens
    if args.enable_tbo:
        num_tokens *= 2
        ttft = max((attn_core_time + attn_other_time) / args.sm_ratio, comm_time1)
        ttft += max((attn_core_time + attn_other_time) / args.sm_ratio, comm_time2)
        ttft += max(moe_time / args.sm_ratio, comm_time1)
        ttft += max(moe_time / args.sm_ratio, comm_time2)
    else:
        ttft = attn_core_time
        ttft += moe_time
        ttft += attn_other_time
        ttft += comm_time1 + comm_time2
    ttft *= config.num_hidden_layers
    ttft *= 1000  # convert to ms
    ttft += 30  # for scheduler

    print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))
    print(
        "{:<40} {:<10.0f}".format(
            "Throughput (TGS:tok/GPU/s):", num_tokens / (ttft / 1000)
        )
    )


def decoding(args, config, gpu, target_bs, kvcache_bytes, avg_context_len):
    print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
    attn = create_attention(config, args.use_fp8_gemm, args.use_fp8_kv)
    attn_core_time = attn.decode_attn_core(
        target_bs, avg_context_len, kvcache_bytes, args.device_type
    )
    attn_other_time = attn.decode_attn_others(target_bs, args.device_type)

    moe = MoE(config, args.use_fp8_gemm)
    moe_time = moe.decode_moe(target_bs, args.device_type, args.world_size)

    comm = Comm(config, gpu, args.world_size, args.num_nodes, args.enable_deepep)
    comm_time1, comm_time2 = comm.decode_comm(target_bs)
    print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
    print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

    num_tokens = target_bs
    if args.enable_tbo:
        num_tokens *= 2
        tpot = max(attn_core_time + attn_other_time, moe_time + comm_time1 + comm_time2)
        tpot *= 2
    else:
        tpot = attn_core_time
        tpot += attn_other_time
        tpot += moe_time
        tpot += comm_time1 + comm_time2
    tpot *= config.num_hidden_layers
    tpot *= 1000  # convert to ms
    tpot += 5  # for scheduler

    print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
    print("{:<40} {:<10.0f}".format("Throughput (TGS):", num_tokens / tpot * 1000))
    if tpot > args.target_tpot:
        print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")


def main(args):
    config = ModelConfig(args.config_path)
    gpu = gpu_map[args.device_type]

    print("\n{s:{c}^{n}}".format(s=" Simulator Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Device type:", args.device_type))
    print("{:<40} {:<10}".format("World size:", args.world_size))
    print("{:<40} {:<10}".format("Attn type:", config.attn_type))
    print("{:<40} {:<10}".format("Use FP8 GEMM:", args.use_fp8_gemm))
    print("{:<40} {:<10}".format("Use FP8 KV:", args.use_fp8_kv))

    print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
    attn_params_bytes = get_attn_params_size(config, args.use_fp8_gemm)
    expert_params_bytes = get_expert_params_size(config, args.use_fp8_gemm)
    print(
        "{:<40} {:<10.2f}".format(
            "One attn params size (MB):", attn_params_bytes / 1024 / 1024
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "One expert params size (MB):", expert_params_bytes / 1024 / 1024
        )
    )
    params_per_gpu = attn_params_bytes + expert_params_bytes * (
        config.num_shared_experts + config.num_routed_experts / args.world_size
    )
    params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
    params_per_gpu *= config.num_hidden_layers
    kvcache_mem = gpu.mem - params_per_gpu - 15 - 5  # 15GB for runtime, 5GB for encoder
    print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))

    print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("KV cache space (GB):", kvcache_mem))
    context_len = args.target_isl + args.target_osl

    if args.decode_bs is None:
        target_bs = math.ceil(args.target_tgs * args.target_tpot / 1000)
    else:
        target_bs = args.decode_bs
    print("{:<40} {:<10}".format("Input seq len:", args.target_isl))
    print("{:<40} {:<10}".format("Output seq len:", args.target_osl))
    print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))
    target_kvcache_bytes = kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len
    kvcache_bytes = get_kvcache_size(config, args.use_fp8_kv)
    print(
        "{:<40} {:<10.2f}".format(
            "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Current per-token KV cache size (KB):", kvcache_bytes / 1024
        )
    )
    if kvcache_bytes > target_kvcache_bytes:
        print("!Error: need smaller kvcache")

    print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))
    print("{:<40} {:<10}".format("Num hidden layers:", config.num_hidden_layers))
    # per-token per-layer gflops
    avg_context_len = int(args.target_isl + args.target_osl / 2)
    attn_core_gflops, other_gflops = get_attn_gflops(
        config, avg_context_len, absorb=True
    )
    moe_gflops = get_moe_gflops(config)
    print(
        "{:<40} {:<10.2f}".format(
            "Per-token per-layer attn core (GFLOPs):", attn_core_gflops
        )
    )
    print(
        "{:<40} {:<10.2f}".format("Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops)
    )
    print(
        "{:<40} {:<10.2f}".format("Per-token per-layer others (GFLOPs):", other_gflops)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Per-token attn core (GFLOPs):", attn_core_gflops * config.num_hidden_layers
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Per-token MoE (GFLOPs):", moe_gflops * config.num_hidden_layers
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Per-token others (GFLOPs):", other_gflops * config.num_hidden_layers
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Per-token total (GFLOPs):",
            (attn_core_gflops + moe_gflops + other_gflops) * config.num_hidden_layers,
        )
    )

    if not args.decode_only:
        prefill(args, config, gpu, kvcache_bytes)

    if not args.prefill_only:
        decoding(args, config, gpu, target_bs, kvcache_bytes, avg_context_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="H20",
        choices=["H20", "H800"],
        help="Device type",
    )
    parser.add_argument("--world-size", type=int, default=1, help="Num of GPUs")
    parser.add_argument("--num-nodes", type=int, default=1, help="Num of nodes")
    parser.add_argument(
        "--max-prefill-tokens", type=int, default=4096, help="Max prefill tokens"
    )
    parser.add_argument(
        "--decode-bs",
        type=int,
        help="Decoding batchsize. If not specified, bs = tgs * tpot.",
    )
    parser.add_argument(
        "--target-tgs", type=float, default=2560, help="Target tokens/s per GPU"
    )
    parser.add_argument("--target-tpot", type=float, default=50, help="TPOT in ms")
    parser.add_argument(
        "--target-isl", type=int, default=4096, help="Input sequence length, in tokens"
    )
    parser.add_argument(
        "--target-osl", type=int, default=2048, help="Output sequence length, in tokens"
    )
    parser.add_argument("--use-fp8-gemm", action="store_true", help="Use fp8 gemm")
    parser.add_argument("--use-fp8-kv", action="store_true", help="Use fp8 kvcache")
    parser.add_argument("--enable-deepep", action="store_true", help="Enable DeepEP")
    parser.add_argument(
        "--enable-tbo", action="store_true", help="Enable two batch overlap"
    )
    parser.add_argument(
        "--sm-ratio",
        type=float,
        default=108 / 132,
        help="In TBO DeepEP normal mode, the SM ratio used for computation",
    )
    parser.add_argument(
        "--prefill-only", action="store_true", help="Only simulate prefill"
    )
    parser.add_argument(
        "--decode-only", action="store_true", help="Only simulate decoding"
    )
    args = parser.parse_args()
    main(args)
