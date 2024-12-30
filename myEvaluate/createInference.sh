#!/bin/bash

models=(
  "Qwen/CodeQwen1.5-7B-Chat"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "NTQAI/Nxcode-CQ-7B-orpo"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "codellama/CodeLlama-7b-Instruct-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
)

n_lst=(
  # 16
  32
  64
)

for model in ${models[@]}
do
    python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate

    for n in ${n_lst[@]}
    do
      # best of n
      python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --n ${n} --temperature 1.0
    done
done