#!/bin/bash

# specific script used solely for debugging codellama greedy decoding result
model="meta-llama/Llama-3.1-8B-Instruct"
python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate
