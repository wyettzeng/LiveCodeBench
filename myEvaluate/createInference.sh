#!/bin/bash
model_name=Qwen/Qwen2.5-Coder-7B-Instruct
python -m lcb_runner.runner.main --model "$model_name" --scenario codegeneration --evaluate

# best of n
model_name=Qwen/Qwen2.5-Coder-7B-Instruct
python -m lcb_runner.runner.main --model "$model_name" --scenario codegeneration --evaluate --n 16 --temperature 1.0
