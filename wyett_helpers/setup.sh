#!/bin/bash

conda create -n livecodebench python=3.10
conda init
conda activate livecodebench
pip install poetry
poetry install --with with-gpu
