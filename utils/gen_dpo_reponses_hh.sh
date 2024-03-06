#!/bin/bash
set -e
# List of temperatures
temps=(0.6 1.0 1.4)

# Function to run jobs in parallel with a limit of 12
run_jobs() {
  local temp="$1"
  local weights_path="$2"
  for idx in 0 500 1000 1500 2000 2500 3000 3500 4000; do
    local end_idx=$((idx + 500))
    [ $idx -eq 4000 ] && end_idx=4500

    echo "Running with temperature $temp"
    srun --gres=gpu:1 -c 12 --mem 128G -p general -w i001,h001,h002,g002 \
      python hh_response_generation.py --temperature $temp --weights_path $weights_path \
      --start_idx $idx --end_idx $end_idx &

    # Track the PID of the background job
    pids+=($!)
    # Wait for some jobs to complete if we've reached the limit of 12
    if [ ${#pids[@]} -eq 12 ]; then
      wait "${pids[@]}"
      pids=()
    fi
  done
}

# Iterate over the temperatures and checkpoints
# for temp in "${temps[@]}"; do
#   run_jobs $temp "/net/scratch/RLHF/HH/pythia28/dpo_jsd_pythia28_beta0.1/LATEST/policy.pt"
# done

# for temp in "${temps[@]}"; do
#   run_jobs $temp "/net/scratch/RLHF/HH/pythia28/dpo_reverse_kl_pythia28_beta0.1/LATEST/policy.pt"
# done

for temp in "${temps[@]}"; do
  run_jobs $temp "/net/scratch/RLHF/HH/pythia28/dpo_forward_kl_pythia28_beta0.1/LATEST/policy.pt"
done

# Wait for any remaining background jobs to complete
wait

