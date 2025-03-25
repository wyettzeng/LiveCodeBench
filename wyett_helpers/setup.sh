#!/bin/bash

conda create -n livecodebench python=3.10
conda init
conda activate livecodebench

uv venv --python 3.11
source .venv/bin/activate

uv pip install -e .

# most models are now newer
pip install transformers --upgrade