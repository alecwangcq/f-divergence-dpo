#!/bin/bash

# Define the path to your python script
PYTHON_SCRIPT_PATH="./dataset_generation.py"

# Loop through chunk indices
for CHUNK_INDEX in {0..9}
do
  # Call your python script with srun and the current chunk index as argument
  srun --gres=gpu:1 -c 8 --mem 80G -p general -w i001,h002 python -u $PYTHON_SCRIPT_PATH $CHUNK_INDEX &
done

# Wait for all background processes to finish
wait
