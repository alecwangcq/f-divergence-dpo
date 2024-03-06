import pandas as pd
import matplotlib.pyplot as plt
# A list to store the most positive and negative response for each prompt
responses = []
diffs = []
# Process each file
for i in range(10):
    # Load the data from the csv file
    data = pd.read_csv(f'imdb_rlhf_{i}.csv')

    # Iterate over the data by 5-row chunks (each chunk corresponds to a prompt)
    for j in range(0, len(data), 5):
        # Extract the chunk corresponding to the current prompt
        chunk = data.iloc[j:j+5]

        # Ensure that all prompts in the chunk are the same
        assert chunk['Prompt'].nunique() == 1, 'All prompts in a chunk should be the same'
        # Get the rows with the highest and lowest reward
        pos = chunk.loc[chunk['Reward'].idxmax()]
        neg = chunk.loc[chunk['Reward'].idxmin()]
        diffs.append(float(pos['Reward']) - float(neg['Reward']))
        # Append to responses
        responses.append({
            'Prompt': chunk['Prompt'].iloc[0],
            'positive_response': pos['Response'],
            'positive_reward': pos['Reward'],
            'negative_response': neg['Response'],
            'negative_reward': neg['Reward']
        })

# Create a DataFrame from the responses list
df = pd.DataFrame(responses)

# Save the DataFrame to a csv file
df.to_csv('result.csv', index=False)
plt.hist(diffs)

plt.savefig('hist.png')
