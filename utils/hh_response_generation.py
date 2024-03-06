import os
os.environ['PYTHONUNBUFFERED'] = 'True'

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from itertools import islice
import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preference_datasets import get_hh

from trlx.pipeline.offline_pipeline import PromptPipeline
import time

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_sequences):
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids, scores, **kwargs):
        for stop_sequence in self.stop_sequences:
            last_tokens = input_ids[-len(stop_sequence):]
            if last_tokens == stop_sequence:
                return True
        return False

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def load_model(model_path, weights_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if weights_path is None:
        return model.eval().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict['state'])
    model = model.to(device)
    model.eval()
    return model


def load_existing_output(output_file):
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    else:
        return pd.DataFrame(columns=["prompt", "original_output", "generated_output"])


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--n_generations", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_cache_dir", type=str, default=".cache/datasets")
    parser.add_argument("--output_dir", type=str, default="hh_responses")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    


    model_path = args.model_path
    weights_path = args.weights_path
    model_name = [x for x in weights_path.split('/') if 'dpo' in x][0]
    tokenizer_name = args.tokenizer_name
    n_generations = args.n_generations
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    batch_size = args.batch_size
    dataset_cache_dir = args.dataset_cache_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.is_available(), torch.cuda.device_count())
    model = load_model(model_path, weights_path, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Model and tokenizer loaded")

    dataset = get_hh(split='test', cache_dir=dataset_cache_dir)
    
    file_name = f"{model_name}-T{temperature}-k{top_k}-p{top_p}-n{n_generations}_seg{args.start_idx}_to_{args.end_idx}.csv"
    output_file_path = os.path.join(args.output_dir, file_name)
    output = load_existing_output(output_file_path)  # Load previously generated responses
    
    eval_prompts = [{"prompt": x, "original_output": y["sft_target"]} for x, y in dataset.items()]

    start_point = len(output) // 25
    eval_prompts = eval_prompts[start_point:args.end_idx * 2]
    
    # truncate to accomdate start_idx and end_idx
    # eval_prompts = eval_prompts[args.start_idx * args.batch_size:args.end_idx * args.batch_size]
    if len(eval_prompts) == 0:
        raise ValueError(f"start_idx {args.start_idx} and end_idx {args.end_idx} are invalid")

    stop_sequences = ["Human:", "human:", "Assistant:", "assistant:", tokenizer.eos_token]
    stop_criteria = CustomStoppingCriteria([tokenizer.encode(x) for x in stop_sequences])
    
    eval_pipeline = PromptPipeline(eval_prompts, 512, tokenizer)
    eval_dataloader = eval_pipeline.create_loader(batch_size)

    begin = time.time()
    # idx = 0
    idx = start_point // batch_size
    # output = pd.DataFrame(columns=["prompt", "original_output", "generated_output"])
    for inputs in eval_dataloader:

        batch_input_ids = inputs["input_ids"].to(device)
        batch_attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            responses =  model.generate(batch_input_ids, 
                                        attention_mask=batch_attention_mask, 
                                        do_sample=True, 
                                        max_new_tokens=512, 
                                        pad_token_id=tokenizer.pad_token_id, 
                                        num_return_sequences=n_generations,
                                        temperature=temperature,
                                        top_p=top_p, 
                                        top_k=top_k,
                                        stopping_criteria=stop_criteria,
                                        return_dict_in_generate=True)

        prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        gen_sequences = responses.sequences[:, batch_input_ids.shape[-1]:]
        batch_generations = tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)
        batch_generations = [x.strip() for x in batch_generations]

        for i in range(batch_size):
            row = pd.DataFrame.from_dict({"prompt": prompts[i], 
                                "original_output": inputs["original_output"][i], 
                                "generated_output": batch_generations[i*n_generations:(i+1)*n_generations]})
            output = pd.concat([output, row], ignore_index=True)
        
        # end = time.time()
        # print(f"Time to generate {batch_size} samples: {end - begin} seconds")

        idx += 1
        if idx % 10 == 0:
            file_name = f"{model_name}-T{temperature}-k{top_k}-p{top_p}-n{n_generations}_seg{args.start_idx}_to_{args.end_idx}.csv"
            output.to_csv(os.path.join(args.output_dir, file_name), index=False)
            # print(f"Generated {idx*batch_size} samples out of {len(dataset)} in {time.time() - begin}s")
        
    output.to_csv(os.path.join(args.output_dir, file_name), index=False)

            
        
    
    
    
    
    
    
    
    