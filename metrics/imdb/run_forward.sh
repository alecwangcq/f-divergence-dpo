
#!/bin/bash

# ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.1_2023-07-11_16-09-44_186553"
# ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.03_2023-07-11_19-24-50_765375"
# ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.01_2023-07-11_22-03-32_088669"
# ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.001_2023-07-11_23-53-52_313895"
# ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.0003_2023-07-12_00-05-28_644490"
# ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.0001_2023-07-12_05-44-17_745326"
# ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.0001_5_epochs_2023-07-12_14-10-48_564156"
ckpt_path=".cache/chaoqi/imdb_dpo_forward_kl_gpt2_large_hh0.0001_10_epochs_2023-07-12_22-41-40_493987"

# an array of directory names
# checkpoints=("step-13440" "step-18240" "step-22080" "step-26880" "step-30720" "step-35520" 
# "step-39360" "step-44160" "step-48000" "step-7680" "LATEST" "step-14400" "step-1920" "step-23040" 
# "step-27840" "step-31680" "step-36480" "step-40320" "step-45120" "step-48960" "step-8640" "step-10560" 
# "step-15360" "step-19200" "step-24000" "step-2880" "step-32640" "step-37440" "step-41280" "step-46080" 
# "step-49920" "step-960" "step-11520" "step-16320" "step-20160" "step-24960" "step-28800" "step-33600" 
# "step-3840" "step-42240" "step-47040" "step-5760" "step-9600" "step-12480" "step-17280" "step-21120" 
# "step-25920" "step-29760" "step-34560" "step-38400" "step-43200" "step-4800" "step-6720")
# checkpoints=("step-960" "step-1920" "step-2880" "step-3840" "step-4800" "step-5760" "step-6720" "step-7680" "step-8640" "step-9600" "step-10560" "step-11520" "step-12480" "step-13440" "step-14400" "step-15360" "step-16320" "step-17280" "step-18240" "step-19200" "step-20160" "step-21120" "step-22080" "step-23040" "step-24000" "step-24960" "step-25920" "step-26880" "step-27840" "step-28800" "step-29760" "step-30720" "step-31680" "step-32640" "step-33600" "step-34560" "step-35520" "step-36480" "step-37440" "step-38400" "step-39360" "step-40320" "step-41280" "step-42240" "step-44160" "step-45120" "step-46080" "step-47040" "step-48000" "step-48960" "step-49920" "LATEST" "LATEST")

# checkpoints=("step-4992" "step-9984" "step-14976" "step-19968" "step-24960" "step-29952" "step-34944" "step-39936" "step-44928" "step-49920" "step-54912" "step-59904" "step-64896" "step-69888" "step-74880" "step-79872" "step-84864" "step-89856" "step-94848" "step-99840" "step-104832" "step-109824" "step-114816" "step-119808" "step-124800" "step-129792" "step-134784" "step-139776" "step-144768" "step-149760" "step-154752" "step-159744" "step-164736" "step-169728" "step-174720" "step-179712" "step-184704" "step-189696" "step-194688" "step-199680" "step-204672" "step-209664" "step-214656" "step-219648" "step-224640" "step-229632" "step-234624" "step-239616" "step-244608" "step-249600" "LATEST" "LATEST")
checkpoints=("step-9984" "step-19968" "step-29952" "step-39936" "step-49920" "step-59904" "step-69888" "step-79872" "step-89856" "step-99840" "step-109824" "step-119808" "step-129792" "step-139776" "step-149760" "step-159744" "step-169728" "step-179712" "step-189696" "step-199680" "step-209664" "step-219648")

# iterate through each directory
for (( i=0; i<${#checkpoints[@]}; i+=1 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
  
  # for example, if you want to list the contents of each directory
  # uncomment the following line
  # ls "$checkpoint"
  # --exclude=g004,g007,g006,g009
done | xargs -I {} -P 15 srun --gres=gpu:1 -c 6 --mem 60G -p general --exclude=g002,g005,g006,g007,g008,g009  python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence forward_kl

# .cache/chaoqi/imdb_dpo_jsd_gpt2_large_hh0.1_2023-07-10_21-39-48_619520/step-16320/policy.pt

# --exclude=g002,g004,g005,g007,g008,g009