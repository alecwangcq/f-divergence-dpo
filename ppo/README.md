To run the experiments, please use

```bash
srun --gres=gpu:1 -c 12 --mem 80G -p general python ppo_sentiment.py --f_divergence forward_kl
```

