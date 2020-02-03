#!/bin/bash

source ~/.profile

export BABYAI_STORAGE=/home/gridsan/jda/code/babyai

MODEL_BASE="BabyAI-SynthLoc-v0_IL_expert_filmcnn_gru_seed1_20-01-17-08-58-09_best"
MODEL_BERT="BabyAI-SynthLoc-v0_IL_expert_filmcnn_gru_seed1_20-01-17-08-57-40_best"
#TURK="demos_turk_test/Batch_clean.csv"
TURK="demos_turk_test/Batch_clean_long.csv"
RESULTS="results_long"

### skyline
#python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" >$RESULTS/eval_orig.txt 2>$RESULTS/eval_orig.err
#
### projection lex only
#python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human --proj "lex" --proj_file sentences.txt >$RESULTS/eval_proj_lex.txt 2>$RESULTS/eval_proj_lex.err
#
## projection bert only
PYTHONHASHSEED=1 python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human --proj "bert" --proj_file sentences.txt >$RESULTS/eval_proj_hier_bert.txt 2>$RESULTS/eval_proj_hier_bert.err

# projection both
#PYTHONHASHSEED=1 python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human --proj "both" --proj_file sentences.txt #>$RESULTS/eval_proj_both.txt 2>$RESULTS/eval_proj_both.err

## LSTM 0
#python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BASE" --turk_file "$TURK" --human >$RESULTS/eval_base.txt 2>$RESULTS/eval_base.err
#
## LSTM+BERT 0
#python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "$MODEL_BERT" --turk_file "$TURK" --human >$RESULTS/eval_bert.txt 2>$RESULTS/eval_bert.err

#for size in 256 512 1024 2048 4096; do
#	# LSTM n
#	python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "ft_${size}_best" --turk_file "$TURK" --human >$RESULTS/eval_base_$size.txt 2>$RESULTS/eval_base_$size.err
#
#	# LSTM+BERT n
#	python -u scripts/evaluate_turk.py --env BabyAI-SynthLoc-v0 --model "ft_b_${size}_best" --turk_file "$TURK" --human >$RESULTS/eval_bert_$size.txt 2>$RESULTS/eval_bert_$size.err
#done
