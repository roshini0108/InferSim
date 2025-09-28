#!/bin/bash

python3 main.py --config-path hf_configs/qwen3-30B-A3B_config.json  \
  --target-tps 2000 --device-type H20 \
  --world-size 4 \
  --decode-only
