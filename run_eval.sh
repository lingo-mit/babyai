#!/bin/bash

source ~/.profile

export BABYAI_STORAGE=/home/gridsan/jda/code/babyai

MODEL_BASE="BabyAI-SynthLoc-v0_IL_expert_filmcnn_gru_seed1_20-01-17-08-58-09_best"
MODEL_BERT="BabyAI-SynthLoc-v0_IL_expert_filmcnn_gru_seed1_20-01-17-08-57-40_best"
#TURK="demos_turk/Batch_clean.csv"
TURK="demos_turk/Batch_clean_long.csv"
RESULTS="results_long"

# skyline
python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" >$RESULTS/eval_orig.txt 2>$RESULTS/eval_orig.err

# baseline
python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human >$RESULTS/eval_base.txt 2>$RESULTS/eval_base.err

# projection lex only
python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human --proj "lex" --proj_file sentences.txt >$RESULTS/eval_proj_lex.txt 2>$RESULTS/eval_proj_lex.err

# projection bert only
python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human --proj "bert" --proj_file sentences.txt >$RESULTS/eval_proj_bert.txt 2>$RESULTS/eval_proj_bert.err

# projection both
python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human --proj "both" --proj_file sentences.txt >$RESULTS/eval_proj_both.txt 2>$RESULTS/eval_proj_both.err

# BERT features
python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BERT" --turk_file "$TURK" --human >$RESULTS/eval_bert.txt 2>$RESULTS/eval_bert.err
