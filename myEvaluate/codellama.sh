#!/bin/bash

# specific script used solely for debugging codellama greedy decoding result
model="codellama/CodeLlama-7b-Instruct-hf"

n_lst=(
#   16
  32
  64
)

# python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --release_version release_v4

for n in ${n_lst[@]}
do
  # best of n
  python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --n ${n} --temperature 1.0 --release_version release_v4
done