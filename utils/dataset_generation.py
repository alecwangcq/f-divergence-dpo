import torch
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


# Loading the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="left"

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

# Generate prompts and responses
generated_prompts_responses = []

# Define batch size
batch_size = 64

file_size = 5000
chunk_index = int(sys.argv[1])

with torch.no_grad():
    for prompt_length in [4]:  # prompt lengths 3-7 tokens (inclusive)
        # Generate prompts from the IMDB dataset
        prompts = [" ".join(review.split()[:prompt_length])+" " for review in imdb_dataset["text"]][chunk_index*file_size:(chunk_index+1)*file_size]
        
        # for i in range(0, len(prompts), batch_size):  # loop through prompts by batch
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            # Prepare the inputs with padding
            input_ids = tokenizer.batch_encode_plus(batch_prompts, padding='longest', return_tensors='pt')

            responses = model.generate(input_ids['input_ids'], do_sample=True, max_length=64, top_p=top_p, num_return_sequences=k, pad_token_id=tokenizer.eos_token_id)
            # Divide responses into chunks of size k (for each original prompt)
            response_chunks = responses.split(k)

            # Prepare batch for sentiment analysis
            sentiment_batch = []

            for j, chunk in enumerate(response_chunks):
                decoded_responses = tokenizer.batch_decode(chunk, skip_special_tokens=True)
                for response in decoded_responses:
                    sentiment_batch.append(response)

            # Prepare sentiment inputs
            sentiment_inputs = sentiment_tokenizer(sentiment_batch, padding='longest', truncation=True, max_length=70, return_tensors='pt')
            # Compute sentiment
            sentiment_outputs = sentiment_model(**sentiment_inputs)
            sentiment_probs = torch.nn.functional.softmax(sentiment_outputs.logits, dim=-1)

            idx = 0
            debug = False
            for j, chunk in enumerate(response_chunks):
                responses = tokenizer.batch_decode(chunk, skip_special_tokens=True)
                for response in responses:
                    if debug:
                        import ipdb; ipdb.set_trace()
                    reward = sentiment_probs[idx, 1].item() - sentiment_probs[idx, 0].item()
                    # if sentiment_outputs.logits[idx, 1] > sentiment_outputs.logits[idx, 0]:
                    #     reward = sentiment_probs[idx, 1].item()
                    # else:
                    #     reward = -sentiment_probs[idx, 0].item()
                    idx += 1
                    # Save prompt, response, and reward
                    generated_prompts_responses.append((batch_prompts[j], response, reward))
            print(batch_prompts[j], response)

# Now generated_prompts_responses contains tuples of (prompt, response, reward)

# Convert the list of tuples into a pandas DataFrame
df = pd.DataFrame(generated_prompts_responses, columns=["Prompt", "Response", "Reward"])

# Save the DataFrame to a CSV file
df.to_csv(f"imdb_rlhf_{chunk_index}.csv", index=False)
