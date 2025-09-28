# InferSim: A lightweight LLM inference simulator

InferSim is a lightweight simulator for LLM inference, writen in pure Python without 3rd-party depenencies. It calculates the TTFT, TPOT and throughput (TGS i.e. tokens/GPU/second) based on computation complexity (FLOPs), GPU computing power (FLOPS), GPU memory bandwidth and MFU (Gathered from SOTA kernel benchmarks). InferSim also estimates the communication latency according to data size and bandwidth.

The main use cases of InferSim include:
- **Model-Sys co-design**: Evaluating the inference perfomance given the model hyperparameters.
- **Inference performance analysis**: Assisting in identifying performance bottlenecks, such as compute-bound and IO-bound scenarios, and supporting optimization efforts.

## Simulation Result

| Model | GPU | Prefill TGS(Actual) | Prefill TGS(Sim) | Decode TGS(Actual) | Decode TGS(Sim) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| DeepSeek-V3 | H800 | 7839 | 9034 | 2324 | 2675 | Actual data from [deepseek/profile-data](https://github.com/deepseek-ai/profile-data/). Simulated with same setup: [example/deepseek-v3/](./example/deepseek-v3/). |
| Qwen3-30B-A3B-BF16 | H20 | 16594 | 17350 | 2749 | 2632 | Actual data tested with SGLang, simulation example: [example/qwen3-30B-A3B/](./example/qwen3-30B-A3B/). |
| Qwen3-8B-FP8 | H20 | 15061 | 16328 | 2682 | 2581 | Actual data tested with SGLang, simulation example: [example/qwen3-8B/](./example/qwen3-8B/). |

## Supported Features

- **Attention**: MHA/GQA, MLA. MFU benchmarks from FlashInfer, FlashAttention-3, FlashMLA.
- **MoE**: GroupedGEMM. MFU benchmarks from DeepGEMM.
- **Linear**: GEMM. MFU benchmarks from DeepGEMM.
- **Parallelization**: DP Attn, EP MoE.
- **Large EP**: DeepEP dispatch and combine, with normal and low_latency mode.

## Help

```
$ python3 main.py --help
usage: main.py [-h] --config-path CONFIG_PATH [--device-type {H20,H800}] [--world-size WORLD_SIZE] [--num-nodes NUM_NODES]
               [--target-tps TARGET_TPS] [--target-tpot TARGET_TPOT] [--target-isl TARGET_ISL] [--target-osl TARGET_OSL]
               [--max-prefill-tokens MAX_PREFILL_TOKENS] [--use-fp8-gemm] [--use-fp8-kv] [--enable-deepep] [--enable-tbo]
               [--sm-ratio SM_RATIO] [--prefill-only] [--decode-only]

optional arguments:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        The path of the hf model config.json
  --device-type {H20,H800}
                        Device type
  --world-size WORLD_SIZE
                        Num of GPUs
  --num-nodes NUM_NODES
                        Num of nodes
  --target-tps TARGET_TPS
                        Target tokens/s per GPU
  --target-tpot TARGET_TPOT
                        TPOT in ms
  --target-isl TARGET_ISL
                        Input sequence length, in tokens
  --target-osl TARGET_OSL
                        Output sequence length, in tokens
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        Max prefill tokens
  --use-fp8-gemm        Use fp8 gemm
  --use-fp8-kv          Use fp8 kvcache
  --enable-deepep       Enable DeepEP
  --enable-tbo          Enable two batch overlap
  --sm-ratio SM_RATIO   In TBO DeepEP normal mode, the SM ratio used for computation
  --prefill-only        Only simulate prefill
  --decode-only         Only simulate decoding
```

## Example

```
$ bash example/qwen3-30B-A3B/decode.sh

================ Simulator Result ================
Device type:                             H20
World size:                              4
Attn type:                               MHA/GQA
Use FP8 GEMM:                            0
Use FP8 KV:                              0
------------------Model Weights-------------------
One attn params size (MB):               36.00
One expert params size (MB):             9.00
Per GPU params size (GB):                15.19
---------------------KV Cache---------------------
KV cache space (GB):                     60.81
Input seq len:                           4096
Output seq len:                          2048
Target decode batchsize:                 100
Target per-token KV cache size (KB):     103.79
Current per-token KV cache size (KB):    96.00
----------------------GFLOPs----------------------
Num hidden layers:                       48
Per-token per-layer attn core:           0.08
Per-token per-layer MoE:                 0.08
Per-token per-layer others:              0.04
Per-token attn core:                     4.03
Per-token MoE:                           3.62
Per-token others:                        1.81
---------------------Decoding---------------------
Attn core MFU:                           0.15
Attn core latency (us):                  361.77
KV loading latency (us):                 298.02
QKV_proj latency (us):                   31.03
O_proj latency (us):                     16.95
Routed experts MFU:                      0.18
Routed experts latency (us):             269.28
Experts loading latency (us):            85.83
Comm before MoE (us):                    4.24
Comm after MoE (us):                     4.24
TPOT (ms):                               38.00
Throughput (TGS):                        2632
```

## Acknowledgement

This work is developed and maintained by Alimama AI Infra Team & Future Living Lab, Alibaba Group.
