#!/bin/bash

python3 main.py --config-path hf_configs/qwen3-8B_config.json \
  --device-type H20 --world-size 1 \
  --max-prefill-tokens 16384 \
  --target-tps 1280 \
  --use-fp8-gemm
