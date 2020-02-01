#!/bin/bash

source ~/.profile

export BABYAI_STORAGE=/home/gridsan/jda/code/babyai

python scripts/make_agent_demos.py --env BabyAI-SynthLoc-v0 --episodes 5000 --valid-episodes 0
