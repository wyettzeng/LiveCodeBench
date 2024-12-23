#!/bin/bash

# NOTE, llama 3.1 seems to need special library version
# vllm==0.5.3.post1
# transformers==4.43.1

# specific script used solely for debugging codellama greedy decoding result
model="meta-llama/Llama-3.1-8B-Instruct"
python -m lcb_runner.runner.main --model $model --scenario codegeneration --evaluate
