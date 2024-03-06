# python gen_model_answer.py --model-path /net/scratch/RLHF/HH/pythia28/dpo_forward_kl_pythia28_beta0.1/LATEST/hf_model --model-id pythia-2.8B-f-dpo-forward-kl-beta01
import glob
import os.path

if __name__ == '__main__':
    # generate shell scripts that can obtain model outputs for GPT4 evaluation
    model_path_pattern = "/path/to/pythia28/*/hf_model"
    model_paths = glob.glob(model_path_pattern)
    output_path = "/path/to/gen_model_answer.sh"
    with open(output_path, 'w') as f:
        # first, write shell header
        f.write("#!/bin/bash\n")
        # second, deactivate the current conda environment
        f.write("conda deactivate\n")
        # third, switch to FastChat dir and activate the conda environment
        f.write("cd /path/to/FastChat\n")
        f.write("conda activate ./env\n")
        # cd to llm_judge dir
        f.write("cd ./fastchat/llm_judge\n")
        # fourth, generate shell commands
        for model_path in model_paths:
            #model_id = model_path.split('/')[-3]
            model_id = model_path.split('/')[-2]
            # replace "." with "_"
            model_id = model_id.replace('.', '_')
            output_file_path = f"/path/to/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/{model_id}.jsonl"
            if not os.path.exists(output_file_path):
                f.write(f"python gen_model_answer.py --model-path {model_path} --model-id {model_id} --num-gpus-per-model 1 --num-gpus-total 4\n")
                # report generating progress
                f.write(f"echo {model_id} generated\n")
    print(f"Shell script generated at {output_path}")

