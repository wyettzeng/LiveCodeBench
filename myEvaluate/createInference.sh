#!/bin/bash

models=(
  # "Qwen/CodeQwen1.5-7B-Chat"
  # "Qwen/Qwen2.5-Coder-7B-Instruct"
  # "NTQAI/Nxcode-CQ-7B-orpo"
  # "meta-llama/Meta-Llama-3-8B-Instruct"
  # "mistralai/Mistral-7B-Instruct-v0.3"
  "codellama/CodeLlama-7b-Instruct-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
)

for model in ${models[@]}
do
    python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate

    # best of n
    python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --n 16 --temperature 1.0
done