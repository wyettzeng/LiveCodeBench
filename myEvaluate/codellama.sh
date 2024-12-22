#!/bin/bash

# specific script used solely for debugging codellama greedy decoding result
model="codellama/CodeLlama-7b-Instruct-hf"
python -m lcb_runner.runner.main --model $model --scenario codegeneration --max_tokens 1024 --evaluate
