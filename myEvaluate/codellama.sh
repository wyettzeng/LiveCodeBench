#!/bin/bash

# specific script used solely for debugging codellama greedy decoding result
model="codellama/CodeLlama-7b-Instruct-hf"
# python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --max_tokens 1024 --evaluate --n 32 --temperature 1.0
# python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate --max_tokens 1024 --evaluate --n 64 --temperature 1.0


# for evaluation
python -m lcb_runner.runner.main --model $model --scenario codegeneration --continue_existing_with_eval --num_process_evaluate 1 --timeout 1 --max_tokens 1024 --evaluate --n 32 --temperature 1.0
# python -m lcb_runner.runner.main --model $model --scenario codegeneration --continue_existing_with_eval --num_process_evaluate 1 --timeout 1 --max_tokens 1024 --evaluate --n 64 --temperature 1.0
