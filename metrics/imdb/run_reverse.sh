
#!/bin/bash

ckpt_path=".cache/chaoqi/imdb_dpo_reverse_kl_gpt2_large_hh0.1_2023-07-11_16-09-44_188031"

# an array of directory names
# checkpoints=("step-13440" "step-18240" "step-22080" "step-26880" "step-30720" "step-35520" 
# "step-39360" "step-44160" "step-48000" "step-7680" "LATEST" "step-14400" "step-1920" "step-23040" 
# "step-27840" "step-31680" "step-36480" "step-40320" "step-45120" "step-48960" "step-8640" "step-10560" 
# "step-15360" "step-19200" "step-24000" "step-2880" "step-32640" "step-37440" "step-41280" "step-46080" 
# "step-49920" "step-960" "step-11520" "step-16320" "step-20160" "step-24960" "step-28800" "step-33600" 
# "step-3840" "step-42240" "step-47040" "step-5760" "step-9600" "step-12480" "step-17280" "step-21120" 
# "step-25920" "step-29760" "step-34560" "step-38400" "step-43200" "step-4800" "step-6720")
checkpoints=("step-960" "step-1920" "step-2880" "step-3840" "step-4800" "step-5760" "step-6720" "step-7680" "step-8640" "step-9600" "step-10560" "step-11520" "step-12480" "step-13440" "step-14400" "step-15360" "step-16320" "step-17280" "step-18240" "step-19200" "step-20160" "step-21120" "step-22080" "step-23040" "step-24000" "step-24960" "step-25920" "step-26880" "step-27840" "step-28800" "step-29760" "step-30720" "step-31680" "step-32640" "step-33600" "step-34560" "step-35520" "step-36480" "step-37440" "step-38400" "step-39360" "step-40320" "step-41280" "step-42240" "step-44160" "step-45120" "step-46080" "step-47040" "step-48000" "step-48960" "step-49920" "LATEST" "LATEST")

# iterate through each directory
for (( i=0; i<${#checkpoints[@]}; i+=2 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
  
  # for example, if you want to list the contents of each directory
  # uncomment the following line
  # ls "$checkpoint"
done | xargs -I {} -P 5 srun --gres=gpu:1 -c 6 --mem 60G -p general --exclude=g002,g005,g006,g007,g008,g009  python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence reverse_kl
