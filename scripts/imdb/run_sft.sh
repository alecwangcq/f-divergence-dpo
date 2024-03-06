# 12:57pm, June 26, 2023
######## repeat training using sft on hh only
ulimit -n 64000; python -u train.py model=gpt2_large datasets=[imdb] loss=sft exp_name=imdb_dpo_gpt2_large gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false