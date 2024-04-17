import torch

# Define a tensor of probabilities
probs = torch.tensor([0.1, 0.2, 0.4, 0.3])

# Number of samples to draw
num_samples = 2

# Sample indices based on the probabilities
sample_indices = torch.multinomial(probs, num_samples, replacement=True)

# Sample values from the tensor
sampled_values = probs[sample_indices]

print("Sampled indices:", sample_indices)
print("Sampled values:", sampled_values)
