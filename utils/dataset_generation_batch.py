import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Loading the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2-large")

# Load the weights from your pre-trained model
model.load_state_dict(torch.load("/net/scratch/RLHF/GPT2-large-IMDB-finetune-1400/gpt2-finetune-imdb-1400.pt"))  # replace with your weights file path

# Load IMDB dataset
imdb_dataset = load_dataset("imdb", split="train+test")

# Define k and top_p for response generation
k = 5  # replace with your preferred number of responses
top_p = 0.95

# Loading the sentiment analysis model and tokenizer
sentiment_model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
sentiment_tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")

# Define batch size
batch_size = 256

# Generate prompts
prompts = [" ".join(review.split()[:4])+" " for review in imdb_dataset["text"]]

def collate_fn(batch):
    input_ids = tokenizer.batch_encode_plus(batch, padding='longest', return_tensors='pt')
    return input_ids['input_ids']

# Data loader for lazy loading
prompt_loader = DataLoader(prompts, batch_size=batch_size, num_workers=8, collate_fn=collate_fn)

# Generate prompts and responses
generated_prompts_responses = []

with torch.no_grad():
    for input_ids in tqdm(prompt_loader):
        responses = model.generate(input_ids, do_sample=True, max_length=64, top_p=top_p, num_return_sequences=k, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the generated ids to strings
        decoded_responses = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in responses]

        # Prepare sentiment inputs
        sentiment_inputs = sentiment_tokenizer(decoded_responses, padding='longest', truncation=True, max_length=70, return_tensors='pt')
        # Compute sentiment
        sentiment_outputs = sentiment_model(**sentiment_inputs)
        sentiment_probs = torch.nn.functional.softmax(sentiment_outputs.logits, dim=-1)

        idx = 0
        debug = False
        for j, response in enumerate(decoded_responses):
            if debug:
                import ipdb; ipdb.set_trace()
            reward = sentiment_probs[idx, 1].item() - sentiment_probs[idx, 0].item()
            idx += 1
            # Save prompt, response, and reward
            generated_prompts_responses.append((prompts[j], response, reward))

# Now generated_prompts_responses contains tuples of (prompt, response, reward)

# Convert the list of tuples into a pandas DataFrame
df = pd.DataFrame(generated_prompts_responses, columns=["Prompt", "Response", "Reward"])

# Save the DataFrame to a CSV file
df.to_csv("imdb_rlhf.csv", index=False)
