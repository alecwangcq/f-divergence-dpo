import torch
from datasets import load_dataset
import datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from tqdm import tqdm
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from preference_datasets import get_hh


def calculate_score(model, input_prompt_ids, input_ids):
    outputs = model(input_prompt_ids)[0].squeeze(0)
    outputs = outputs.log_softmax(-1)  # logits to log probs

    outputs = outputs[input_ids.shape[-1] - 1: -1, :]
    prompt_ids = input_prompt_ids[0, input_ids.shape[-1]:]
    log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]

    score = log_probs.sum().item() 

    return score, len(log_probs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--ref_checkpoint', type=str, default='/net/scratch/RLHF/HH/pythia28/SFT/LATEST/policy.pt')
    parser.add_argument('--rootdir', type=str, default='/home/yiboj/hh-new')
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--cache_dir', type=str, default="./")

    args = parser.parse_args()
    parent_dir = os.path.dirname(args.checkpoint)
    step_idxs = args.checkpoint.split('/')[-2]
    model_name = args.checkpoint.split('/')[-3]
    beta = float(model_name.split('beta')[1])

    root_dit = args.rootdir
    data_path = args.data_path

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b')
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    
    # load trained model
    state_dict_path = args.checkpoint
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-2.8b')
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu'))['state'])
    # import ipdb; ipdb.set_trace()
    model.to('cuda')

    # load reference model
    ref_state_dict_path = args.ref_checkpoint
    ref_model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-2.8b')
    ref_model.load_state_dict(torch.load(ref_state_dict_path, map_location=torch.device('cpu'))['state'])
    ref_model.to('cuda')

    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split="test", cache_dir=args.cache_dir)

    columns = ['chosen_reward', 'chosen_reject', 'pred', 'prob', 'chosen_score', 'reject_score', 'chosen_ref_score', 'reject_ref_score', 'chosen_length', 'reject_length']
    data_frame = pd.DataFrame(columns=columns)

    with torch.no_grad():
        for prompt, data in get_hh(split="test", cache_dir=args.cache_dir).items():
            rewards= []
            scores = []
            ref_scores = []
            lengths = []

            for index in range(2):
                prompt_with_answer = prompt + data['responses'][index]

                input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)["input_ids"].to(model.device)
                input_prompt_ids = tokenizer(prompt_with_answer, return_tensors='pt', truncation=True, padding=True)["input_ids"].to(model.device)

                score, length = calculate_score(model, input_prompt_ids, input_ids)
                ref_score, length = calculate_score(ref_model, input_prompt_ids, input_ids)

                reward = beta * (score - ref_score)
                rewards.append(reward)

                scores.append(score)
                ref_scores.append(ref_score)
                lengths.append(length) 

            pred = rewards[0] > rewards[1]
            prob = 1./(1. + np.exp(rewards[1] - rewards[0]))

            new_row = {'chosen_reward': rewards[0], 'chosen_reject': rewards[1], 'pred': pred, 'prob':prob, 'chosen_score': scores[0], 'reject_score': scores[1], 'chosen_ref_score': ref_scores[0], 'reject_ref_score': ref_scores[1], 'chosen_length': lengths[0], 'reject_length': lengths[1]}
            data_frame = pd.concat([data_frame, pd.DataFrame([new_row])], ignore_index=True)

    test_model_path = os.path.join(root_dit, model_name)
    os.makedirs(test_model_path, exist_ok=True)
    test_path = os.path.join(test_model_path, step_idxs)
    os.makedirs(test_path, exist_ok=True)
    samples_path = os.path.join(test_path, "test.jsonl")   
    data_frame[['chosen_reward', 'chosen_reject', 'pred', 'prob', 'chosen_score', 'reject_score', 'chosen_ref_score', 'reject_ref_score', 'chosen_length', 'reject_length']].to_json(samples_path, lines=True, orient="records")  
