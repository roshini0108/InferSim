#!/bin/bash

python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --target-tps 1280 --device-type H800 \
  --world-size 32 --num-nodes 4 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --max-prefill-tokens 8192 --prefill-only
