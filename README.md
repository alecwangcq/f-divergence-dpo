# $f$-DPO
This repository contains the implementation of $f$ - DPO (Direct Preference Optimization) with various divergence regularizations, such as forward KL, reverse KL, Jensen-Shannon divergence, and $\alpha$-divergences ( $\alpha \in (0,1)$ ). The code is designed for producing the results presented in our ICLR 2024 paper, "[Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints](https://arxiv.org/pdf/2309.16240.pdf)".

**Update (09/13/2024)**: The [TRL](https://github.com/huggingface/trl) library now also supports DPO with $f$-divergences. Check out the documentation for more details [doc](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig.f_divergence_type).

# Requirements
To get started, please install the required libraries first:
```
ipykernel==6.23.1
numpy==1.24.3
tokenizers==0.13.3
torch==2.0.1
tqdm==4.65.0
transformers==4.29.2
datasets==2.12.0
beautifulsoup4==4.12.2
wandb==0.15.3
hydra-core==1.3.2
tensor-parallel==1.2.4
```

# How to Run?
For experiments on the PPO with the IMDB-sentiment dataset, refer to the scripts in the "ppo/scripts/" folder. For example:

```
cd ppo
bash scripts/sweep_jsd.sh
```

For PPO on the Anthropic HH dataset, check "run_ppo_hh.sh" in the "ppo/" folder. For instance:
```
cd ppo
bash run_ppo_hh.sh  # Ensure you perform sft before PPO fine-tuning.
```

For $f$-DPO on the IMDB-sentiment or Anthropic HH dataset, the scripts are located in the 'scripts' folder. For example:
```
bash scripts/hh/run_jsd.sh  # Ensure sft is done before $f$-DPO fine-tuning.
bash scripts/imdb/run_jsd.sh
```

For the mt-bench evaluation, consult the README file in the "mt_bench/" folder.

For calibration experiments, refer to the README file in the "cali/" folder.

# Citing Our Work
If our work assists in your research, kindly cite it as follows:
```
@inproceedings{
wang2024beyond,
title={Beyond Reverse {KL}: Generalizing Direct Preference Optimization with Diverse Divergence Constraints},
author={Chaoqi Wang and Yibo Jiang and Chenghao Yang and Han Liu and Yuxin Chen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=2cRzmWXK9N}
}
```

# Acknowledgements
Our code builds upon these codebases:
- [DPO: Direct Preference Optimization](https://github.com/eric-mitchell/direct-preference-optimization)
- [Transformer Reinforcement Learning X (TRLX)](https://github.com/CarperAI/trlx/tree/main)
