# 12:57pm, June 26, 2023
######## training for dpo using jsd but sft is done on hh and shp. #######
# ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=dpo fsdp_port=12355 loss.divergence=alpha_divergence loss.beta=0.1 loss.alpha=0.3 exp_name=anthropic_dpo_alpha_divergence_pythia28_hh_beta0.1_alpha0.3 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=.cache/chaoqi/anthropic_dpo_pythia28_2023-06-26_13-19-27_080816/LATEST/policy.pt
ulimit -n 64000; python -u train.py model=pythia28 datasets=[hh] loss=dpo fsdp_port=12355 loss.divergence=alpha_divergence loss.beta=0.3 loss.alpha=0.3 exp_name=anthropic_dpo_alpha_divergence_pythia28_hh_beta0.3_alpha0.3 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=.cache/chaoqi/anthropic_dpo_pythia28_2023-06-26_13-19-27_080816/LATEST/policy.pt

