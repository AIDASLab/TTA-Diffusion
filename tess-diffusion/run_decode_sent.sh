#!/bin/bash

# TOKEN_ALPHA_VALUES=(0.2 0.4 0.6 0.8 1.0)
# STEPS=(50 100 200)
# ITERS=(3)

python controlled_decode_sentiment_multi.py \
  --control_lambda=2000 \
  --token_schedule="adaptive_grad" \
  --token_alpha=0.6 \
  --token_beta=0.0 \
  --num_generate_samples=50 \
  --top_p=0.99 \
  --sub_classifiers="cardiffnlp/twitter-roberta-base-sentiment-latest" \
  --num_inference_diffusion_steps=200 \
  --classifier_begin_step=200 \
  --classifier_end_step=0 \
  --eval_per_n_step=100 \
  --schedule_warm_up_step=0 \
  --control_iteration=3 \
  --fluency_classifier_begin_step=0 \
  --fluency_classifier_end_step=0 \
  --fluency_classifier_lambda=0 \
  --fluency_classifier_iteration=0 \
  --output_dir='./results/imdb_sentiment_c4_penalty_200' \
  --checkpoint_dir='./${LOCAL_DIR}/outputs/paper_experiments/c4_200' \
  --checkpoint_file='checkpoint-40000' \
  # --ddim=True 