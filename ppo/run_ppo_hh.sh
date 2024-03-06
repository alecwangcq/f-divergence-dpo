# srun --gres=gpu:4 -c 24 --mem 256G -p general --exclude g006 -w h002 accelerate launch --num_processes 3  --config_file configs/hh_config.yaml ppo_hh.py --f_divergence reverse_kl  --model_archive ../.cache/chaoqi/anthropic_dpo_pythia28_2023-06-26_13-19-27_080816/LATEST/policy.pt

#!/bin/bash
set -e

# init_kl_coefs=(0.03 0.1 0.3)
# init_kl_coefs=(0.1)

# alphas=(0.3 0.5 0.7)


srun --gres=gpu:4 -c 24 --mem 256G -p general --exclude g006 -w h002 accelerate launch --num_processes 3  --config_file configs/hh_config.yaml ppo_hh.py --f_divergence alpha_divergence --alpha 0.3  --model_archive ../.cache/chaoqi/anthropic_dpo_pythia28_2023-06-26_13-19-27_080816/LATEST/policy.pt &

srun --gres=gpu:4 -c 24 --mem 256G -p general --exclude g006 -w h001 accelerate launch --num_processes 3  --config_file configs/hh_config.yaml ppo_hh.py --f_divergence alpha_divergence --alpha 0.5  --model_archive ../.cache/chaoqi/anthropic_dpo_pythia28_2023-06-26_13-19-27_080816/LATEST/policy.pt &

srun --gres=gpu:4 -c 24 --mem 256G -p general --exclude g006 -w i001 accelerate launch --num_processes 3  --config_file configs/hh_config.yaml ppo_hh.py --f_divergence alpha_divergence --alpha 0.7  --model_archive ../.cache/chaoqi/anthropic_dpo_pythia28_2023-06-26_13-19-27_080816/LATEST/policy.pt &
