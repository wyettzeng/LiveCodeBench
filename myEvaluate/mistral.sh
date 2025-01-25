#!/bin/bash

model="mistralai/Mistral-7B-Instruct-v0.3"

n_lst=(
  16
  32
  64
)

python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --release_version release_v4

for n in ${n_lst[@]}
do
  # best of n
  python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --n ${n} --temperature 1.0 --release_version release_v4
done