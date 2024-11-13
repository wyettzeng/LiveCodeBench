#!/bin/bash
model_name=Qwen/Qwen2.5-Coder-7B-Instruct
python -m lcb_runner.runner.main --model ${model_name} --scenario codegeneration --evaluate --trust_remote_code --not_fast

# best of n
python -m lcb_runner.runner.main --model ${model_name} --scenario codegeneration --evaluate --trust_remote_code --not_fast --n 16 --temperature 1.0
