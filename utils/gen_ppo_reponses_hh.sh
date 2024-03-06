#!/bin/bash
set -e
# List of temperatures
# temps=(0.6 1.0 1.4)

# # Iterate over the temperatures
# for temp in ${temps[@]}; do
#   echo "Running with temperature $temp"
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w i001  python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_jsd_0.1 --start_idx 0 --end_idx 1000 &
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w i001   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_jsd_0.1 --start_idx 1000 --end_idx 2000&
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w i001   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_jsd_0.1 --start_idx 2000 --end_idx 3000 &
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w i001   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_jsd_0.1 --start_idx 3000 --end_idx 5000 &
# done

# for temp in ${temps[@]}; do
#   echo "Running with temperature $temp"
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w h001   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_reverse_kl_0.1 --start_idx 0 --end_idx 1000 &
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w h001  python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_reverse_kl_0.1 --start_idx 1000 --end_idx 2000 &
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w h001   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_reverse_kl_0.1 --start_idx 2000 --end_idx 3000 &
#   srun --gres=gpu:1 -c 12 --mem 128G -p general -w h001   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_reverse_kl_0.1 --start_idx 3000 --end_idx 5000 &
# done

temps=(1.4)
for temp in ${temps[@]}; do
  echo "Running with temperature $temp"
  srun --gres=gpu:1 -c 12 --mem 128G -p general -w h002   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_forward_kl_0.1 --start_idx 0 --end_idx 1000 &
  srun --gres=gpu:1 -c 12 --mem 128G -p general -w h002   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_forward_kl_0.1 --start_idx 1000 --end_idx 2000 &
  srun --gres=gpu:1 -c 12 --mem 128G -p general -w h002   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_forward_kl_0.1 --start_idx 2000 --end_idx 3000 &
  srun --gres=gpu:1 -c 12 --mem 128G -p general -w h002   python hh_response_generation_ppo.py --temperature $temp --ckpt_path /home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_forward_kl_0.1 --start_idx 3000 --end_idx 5000 &
done


#!/bin/bash
# set -e
# # List of temperatures
# temps=(0.6 1.0 1.4)

# Function to run jobs in parallel with a limit of 12
# run_jobs() {
#   local temp="$1"
#   local ckpt_path="$2"
#   for idx in 0 1000 2000 3000; do
#     local end_idx=$((idx + 1000))
#     [ $idx -eq 3000 ] && end_idx=5000

#     echo "Running with temperature $temp"
#     srun --gres=gpu:1 -c 12 --mem 128G -p general -w h001,h002,i001 \
#       python hh_response_generation_ppo.py --temperature $temp --ckpt_path $ckpt_path \
#       --start_idx $idx --end_idx $end_idx &

#     # Track the PID of the background job
#     pids+=($!)
#     # Wait for some jobs to complete if we've reached the limit of 12
#     if [ ${#pids[@]} -eq 12 ]; then
#       wait "${pids[@]}"
#       pids=()
#     fi
#   done
# }

# # Iterate over the temperatures and checkpoints
# for temp in "${temps[@]}"; do
#   run_jobs $temp "/home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_jsd_0.1"
# done

# for temp in "${temps[@]}"; do
#   run_jobs $temp "/home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_reverse_kl_0.1"
# done

# for temp in "${temps[@]}"; do
#   run_jobs $temp "/home/chaoqi/projects/weights/temp/checkpoint/ppo_hh_forward_kl_0.1"
# done

# Wait for any remaining background jobs to complete
# wait

