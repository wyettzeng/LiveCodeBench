#!/bin/bash

models=(
  "Qwen/CodeQwen1.5-7B-Chat"
  # "Qwen/Qwen2.5-Coder-7B-Instruct"
  "NTQAI/Nxcode-CQ-7B-orpo"
  "meta-llama/Meta-Llama-3-8B-Instruct"
)

for model in ${models[@]}
do
    # python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate

    # best of n
    python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --n 16 --temperature 1.0
done