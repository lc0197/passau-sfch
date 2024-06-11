#!/usr/bin/env bash
#!/bin/sh

# AUDIO

python3 run_rnn_experiments.py --features wav2vec --normalize 0 --name humor_wav2vec --task humor --rnn_hidden_dims 128 --rnn_num_layers 2 --directions bi --lr 0.001 --rnn_dropout 0. --num_seeds 5  --save_checkpoints  --save_predictions

python3 run_rnn_experiments.py --features egemaps --normalize 1 --name humor_egemaps --task humor --rnn_hidden_dims 64 --rnn_num_layers 2 --directions bi --lr 0.005 --rnn_dropout 0. --num_seeds 5  --save_checkpoints --save_predictions

python3 run_rnn_experiments.py --features ds --normalize 0 --name humor_ds --task humor --rnn_hidden_dims 64 --rnn_num_layers 4 --directions bi --lr 0.001 --rnn_dropout 0. --num_seeds 5  --save_checkpoints  --save_predictions

# VIDEO

python3 run_rnn_experiments.py --features farl --normalize 0 --name humor_farl --task humor --rnn_hidden_dims 128 --rnn_num_layers 4 --directions uni --lr 0.0005 --rnn_dropout 0.2 --num_seeds 5 --save_checkpoints --save_predictions

python3 run_rnn_experiments.py --features faus --normalize 0 --name humor_faus --task humor --rnn_hidden_dims 128 --rnn_num_layers 2 --directions bi --lr 0.00005 --rnn_dropout 0.2 --num_seeds 5 --save_checkpoints --save_predictions

python3 run_rnn_experiments.py --features vggface2 --normalize 0 --name humor_vggface2 --task humor --rnn_hidden_dims 128 --rnn_num_layers 2 --directions bi --lr 0.0001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions

# TEXT

python3 run_rnn_experiments.py --features electra --normalize 0 --name humor_electra --task humor --rnn_hidden_dims 128 --rnn_num_layers 4 --directions bi --lr 0.001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions

python3 run_rnn_experiments.py --features bert --normalize 0 --name humor_bert --task humor --rnn_hidden_dims 64 --rnn_num_layers 4 --directions uni --lr 0.001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions

python run_rnn_experiments.py --features sentiment-bert --normalize 0 --name humor_sentiment_bert --task humor --rnn_hidden_dims 256 --rnn_num_layers 2 --directions uni --lr 0.001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions

# MULTIMODAL

python3 run_trf_experiments.py  --model_type mult --features_v farl --features_a wav2vec --features_t electra --normalize 0 0 0 --trf_num_heads 8 --trf_num_v_layers 1 --trf_num_at_layers 1 --lr 0.001 --regularization 0. --train_batch_size 8 --epochs 15 --patience 3 --num_seeds 5 --trf_model_dim 64 --context_window 8 --trf_pos_emb sinus

python3 run_trf_experiments.py --model_type v_focused --features_v farl --features_a wav2vec --features_t electra --normalize 0 0 0 --trf_num_heads 8 --trf_num_v_layers 0 --trf_num_at_layers 0 --trf_num_mm_layers 0 --trf_pos_emb sinus --lr 0.001 --regularization 0. --train_batch_size 8 --epochs 15 --patience 3 --num_seeds 5 --trf_model_dim 32 --context_window 8
