#!/bin/bash
set -e

# init_kl_coefs=(0.03 0.1 0.3)
init_kl_coefs=(0.03)

alphas=(0.3 0.5 0.7)

for alpha in "${alphas[@]}"
do
  for init_kl_coef in "${init_kl_coefs[@]}"
  do
    srun --gres=gpu:1 -c 8 --mem 64G -p general --exclude=g006 python ppo_sentiment.py --f_divergence alpha_divergence --init_kl_coef $init_kl_coef --alpha $alpha --kl_in_reward True &
  done
#   wait
done
