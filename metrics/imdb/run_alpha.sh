#!/bin/bash

ckpt_path=".cache/chaoqi/imdb_dpo_alpha_divergence_0.1_gpt2_large_hh0.01_10_epochs_2023-07-14_12-03-10_096562       "
 
# checkpoints=("step-4992"  "step-4992" "step-9984" "step-9984" "step-14976" "step-19968" "step-24960" "step-29952" "step-34944" "step-39936" "step-44928" "step-49920" "step-54912" "step-59904" "step-64896" "step-69888" "step-74880" "step-79872" "step-84864" "step-89856" "step-94848" "step-99840" "step-104832" "step-109824" "step-114816" "step-119808" "step-124800" "step-129792" "step-134784" "step-139776" "step-144768" "step-149760" "step-154752" "step-159744" "step-164736" "step-169728" "step-174720" "step-179712" "step-184704" "step-189696" "step-194688" "step-199680" "step-204672" "step-209664" "step-214656" "step-219648" "step-224640" "step-229632" "step-234624" "step-239616" "step-244608" "step-249600" "LATEST" "LATEST")

checkpoints=("step-4992" "step-4992" "step-9984" "step-9984" "step-14976" "step-19968" "step-24960" "step-29952" "step-34944" "step-39936" "step-44928" "step-49920" "step-54912" "step-59904" "step-64896" "step-69888" "step-74880" "step-79872" "step-84864" "step-89856" "step-94848" "step-99840" "step-104832" "step-109824" "step-114816" "step-119808" "step-124800" "step-129792" "step-134784" "step-139776" "step-144768" "step-149760" "step-154752" "step-159744" "step-164736" "step-169728" "step-174720" "step-179712" "step-184704" "step-189696" "step-194688" "step-199680" "step-204672" "step-209664" "step-214656" "step-219648" "step-224640" "step-229632" "step-234624" "step-239616" "step-244608" "step-249600" "step-254592" "step-259584" "step-264576" "step-269568" "step-274560" "step-279552" "step-284544" "step-289536" "step-294528" "step-299520" "step-304512" "step-309504" "step-314496" "step-319488" "step-324480" "step-329472" "step-334464" "step-339456" "step-344448" "step-349440" "step-354432" "step-359424" "step-364416" "step-369408" "step-374400" "step-379392" "step-384384" "step-389376" "step-394368" "step-399360" "step-404352" "step-409344" "step-414336" "step-419328" "step-424320" "step-429312" "step-434304" "step-439296" "step-444288" "step-449280" "step-454272" "step-459264" "step-464256" "step-469248" "step-474240" "step-479232"  "step-489216"  "step-499200")


# iterate through each directory
for (( i=0; i<${#checkpoints[@]}; i+=2 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
done | xargs -I {} -P 25 srun --gres=gpu:1 -c 6 --mem 60G -p general  --exclude=g002,g005,g006,g007,g008,g009   python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence alpha_divergence --alpha 0.1


ckpt_path='.cache/chaoqi/imdb_dpo_alpha_divergence_0.3_gpt2_large_hh0.01_10_epochs_2023-07-14_12-21-44_404353'
for (( i=0; i<${#checkpoints[@]}; i+=2 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
done | xargs -I {} -P 25 srun --gres=gpu:1 -c 6 --mem 60G -p general  --exclude=g002,g005,g006,g007,g008,g009   python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence alpha_divergence --alpha 0.3

ckpt_path='.cache/chaoqi/imdb_dpo_alpha_divergence_0.5_gpt2_large_hh0.01_10_epochs_2023-07-14_12-03-10_093799'
for (( i=0; i<${#checkpoints[@]}; i+=2 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
done | xargs -I {} -P 25 srun --gres=gpu:1 -c 6 --mem 60G -p general  --exclude=g002,g005,g006,g007,g008,g009   python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence alpha_divergence --alpha 0.5

ckpt_path='.cache/chaoqi/imdb_dpo_alpha_divergence_0.7_gpt2_large_hh0.01_10_epochs_2023-07-14_12-21-44_406113'
for (( i=0; i<${#checkpoints[@]}; i+=2 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
done | xargs -I {} -P 25 srun --gres=gpu:1 -c 6 --mem 60G -p general  --exclude=g002,g005,g006,g007,g008,g009   python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence alpha_divergence --alpha 0.7

ckpt_path='.cache/chaoqi/imdb_dpo_alpha_divergence_0.9_gpt2_large_hh0.01_10_epochs_2023-07-14_12-29-13_579851'
for (( i=0; i<${#checkpoints[@]}; i+=2 ))
do
  checkpoint=${checkpoints[$i]}
  # perform operations on $checkpoint
  echo "$checkpoint"
done | xargs -I {} -P 25 srun --gres=gpu:1 -c 6 --mem 60G -p general  --exclude=g002,g005,g006,g007,g008,g009   python metrics/imdb/imdb_eval_metrics.py --checkpoint $ckpt_path/{}/policy.pt --divergence alpha_divergence --alpha 0.9
