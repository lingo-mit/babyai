#!/bin/bash

source ~/.profile

export BABYAI_STORAGE=/home/gridsan/jda/code/babyai
#MODEL=BabyAI-SynthLoc-v0_IL_expert_filmcnn_gru_seed1_20-01-17-08-57-40_best
MODEL=BabyAI-SynthLoc-v0_IL_expert_filmcnn_gru_seed1_20-01-17-08-58-09_best
TURK=Batch_clean.csv

scripts/train_il.py --env BabyAI-SynthLoc-v0 --demos BabyAI-SynthLoc-v0_agent --batch-size 64 --epochs 10 --pretrained-model $MODEL --use-nl-demos $TURK --episodes 256 --model ft_b_256

scripts/train_il.py --env BabyAI-SynthLoc-v0 --demos BabyAI-SynthLoc-v0_agent --batch-size 64 --epochs 10 --pretrained-model $MODEL --use-nl-demos $TURK --episodes 512 --model ft_b_512

scripts/train_il.py --env BabyAI-SynthLoc-v0 --demos BabyAI-SynthLoc-v0_agent --batch-size 64 --epochs 10 --pretrained-model $MODEL --use-nl-demos $TURK --episodes 1024 --model ft_b_1024

scripts/train_il.py --env BabyAI-SynthLoc-v0 --demos BabyAI-SynthLoc-v0_agent --batch-size 64 --epochs 10 --pretrained-model $MODEL --use-nl-demos $TURK --episodes 2048 --model ft_b_2048

scripts/train_il.py --env BabyAI-SynthLoc-v0 --demos BabyAI-SynthLoc-v0_agent --batch-size 64 --epochs 10 --pretrained-model $MODEL --use-nl-demos $TURK --episodes 4096 --model ft_b_4096
