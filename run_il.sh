#!/bin/bash

source ~/.profile

export BABYAI_STORAGE=/home/gridsan/jda/code/babyai

scripts/train_il.py --env BabyAI-SynthLoc-v0 --demos BabyAI-SynthLoc-v0 --batch-size 256 --epochs 3 --bert 1
