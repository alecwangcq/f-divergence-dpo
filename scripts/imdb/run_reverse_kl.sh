# 12:57pm, June 26, 2023
######## training for dpo using reverse_kl but sft is done on hh and shp. ########
# model.archive=.cache/chaoqi/imdb_dpo_gpt2_large_2023-07-10_16-45-15_446529/LATEST/policy.pt
ulimit -n 64000; python -u train.py model=gpt2_large datasets=[imdb] loss=dpo fsdp_port=49155  loss.divergence=reverse_kl loss.beta=0.1 exp_name=imdb_dpo_reverse_kl_gpt2_large_hh0.1 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16  eval_every=1000