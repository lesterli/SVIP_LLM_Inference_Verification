import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
import torch.nn.functional as F

def get_model_dimension(model_name, json_file_path='./dim_config.json'):
    try:
        with open(json_file_path, 'r') as json_file:
            dim_config = json.load(json_file)
    except FileNotFoundError:
        return "JSON file not found"
    return dim_config.get(model_name, "Model name not found")

def extract_subsequences(input_ids_batch, n):
    subsequences_count = Counter()
    for input_ids in input_ids_batch.tolist():
        for i in range(len(input_ids) - n + 1):
            subsequence = tuple(input_ids[i:i + n])
            subsequences_count[subsequence] += 1
    return subsequences_count


def sample_secrets(subsequences_count, m, temperature=1.0):

    subsequences = list(subsequences_count.keys())
    frequencies = list(subsequences_count.values())

    # Apply temperature scaling to the frequencies
    scaled_frequencies = [f ** (1.0 / temperature) for f in frequencies]

    # Normalize to get probabilities
    total = sum(scaled_frequencies)
    probabilities = [f / total for f in scaled_frequencies]

    # Sample subsequences based on the computed probabilities
    sampled_secrets = random.choices(subsequences, weights=probabilities, k=m)

    return [list(seq) for seq in sampled_secrets]


def generate_labels(secrets, input_ids_batch):
    labels = []
    for secret in secrets:
        secret_len = len(secret)
        labels_for_secret = [1 if any(secret == input_ids[i:i + secret_len] for i in range(len(input_ids) - secret_len + 1)) else 0 for input_ids in input_ids_batch.tolist()]
        labels.append(labels_for_secret)
    return torch.tensor(labels).float().T 

def pairwise_distance(tensor):
    distance_matrix = torch.cdist(tensor, tensor, p=2)
    return distance_matrix


def distance_consistency_loss(A, B):
    dist_A = pairwise_distance(A)
    dist_B = pairwise_distance(B)
    
    dist_A_normalized = F.normalize(dist_A, p=2, dim=-1)
    dist_B_normalized = F.normalize(dist_B, p=2, dim=-1)
    
    loss = F.mse_loss(dist_A_normalized, dist_B_normalized)
    
    return loss

def distance_margin_loss(A, B, margin=1.0):
    distance = F.pairwise_distance(A, B, p=2)
    loss = F.relu(margin - distance)
    return loss.mean()

def contrastive_batch_loss(mlp_outputs, margin=1.0):
    mlp_outputs=mlp_outputs.squeeze(1)
    distances = torch.cdist(mlp_outputs, mlp_outputs, p=2)

    positive_loss = F.relu(margin - distances).triu(diagonal=1)

    total_loss = positive_loss.sum() / (mlp_outputs.size(0) * (mlp_outputs.size(0) - 1) / 2)

    return total_loss