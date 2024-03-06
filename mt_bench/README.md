# Reproduce MT-Bench Results in the paper
### Author: Chenghao Yang (chenghao@uchicago.edu)

## Requirements
Please install [FastChat](https://github.com/lm-sys/FastChat) from source (clone-and-install) as our codes depends on the evaluation codes in `FastChat/fastchat/llm_judge` package.

We have uploaded the evaluation codes we used to backup the version we used in our paper. This includes `gen_model_answer.py`, `gen_judgment.py`, `show_result.py`, `compute_agreement.py` in this folder. If you come across any dependency problem, perhaps you can try to copy-paste the corresponding files in this folder to the path where you install FastChat from source.

## Convert Trained Model Checkpoint to Huggingface Format
It is possible that the fine-tuned model checkpoint (usually named as `policy.pt`) does not conform with the standard huggingface checkpoint folder format (esp. for tokenizer part), therefore we need to first convert the checkpoint format. 

Check out `convert_dpo_trainer_file_to_huggingface.py` for more details and change the path to your fine-tuned model checkpoint path.

## Generate Model Responses for MT-Bench
Check out `generate_model_outputs_for_gpt4_eval.py` -- this file will help you generate bash script to run model generation for your fine-tuned checkpoints in a batched way.

## Run GPT-4 Eval
We prepare an example bash script to run MT-Bench for pairwise comparison using GPT-4 as the evaluator. Check out the script `run_gen_gpt4_judge_pairwise.sh`, fill in your OPENAI API Key and uncomment those `gen_jugement.py` lines to run MT-Bench based on GPT-4 Eval.

## Check GPT-4 Eval Competition Samples
Check out `export_gpt4_eval_competition_samples.py` and configure the path there.
