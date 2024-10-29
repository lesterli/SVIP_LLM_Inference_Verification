import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import yaml
from models import *
from mydatasets import *
from utils import *
import argparse

class InverseMLP(nn.Module):
    def __init__(self, output_dim, secret_dim):
        super(InverseMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, secret_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)

def load_model(model_path, secret_dim, output_dim,sentence_model_name,sentence_embed_dim, device):
    model = LearnableYModel(secret_dim=secret_dim, output_dim=output_dim,sentence_model_name=sentence_model_name,sentence_embed_dim=sentence_embed_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def optimize_mlp_input(f_function, output2, device, num_steps=100, lr=0.01):
    loss_fn = nn.MSELoss()
    f_function.eval()

    mlp_input = torch.randn((output2.size(0), 1024), device=device, requires_grad=True)
    optimizer = optim.Adam([mlp_input], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        output_f = f_function(mlp_input)
        loss = loss_fn(output_f, output2)
        loss.backward()
        optimizer.step()

    optimized_output_f = f_function(mlp_input)
    
    return optimized_output_f

def load_model_and_extract(model_path, model_config, device):
    model = TransformerGFWithSecret(
        input_dim=model_config['model']['input_dim'], 
        output_dim=model_config['model']['output_dim'], 
        secret_dim=model_config['model']['secret_dim'], 
        num_layers=model_config['model']['num_layers'], 
        num_heads=model_config['model']['num_heads']
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    secret_encoder = model.secret_encoder
    f_function = model.f
    return secret_encoder, f_function

def generate_training_data(secret_encoder, secret_dim, num_samples=10000, batch_size=256, device='cpu'):
    x_samples = []
    y_samples = []
    secret_encoder.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            x_batch = torch.randint(0, 2, (batch_size, secret_dim)).float().to(device)
            y_batch = secret_encoder(x_batch).cpu()
            x_batch = x_batch.cpu()
            x_samples.append(x_batch)
            y_samples.append(y_batch)
    x_samples = torch.cat(x_samples, dim=0)
    y_samples = torch.cat(y_samples, dim=0)
    return y_samples, x_samples

def train_inverse_model(secret_encoder, inverse_mlp, secret_dim, epochs=10, batch_size=256, num_samples=10000, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(inverse_mlp.parameters(), lr=0.001)
    y_samples, x_samples = generate_training_data(secret_encoder, secret_dim, num_samples, device=device)
    for epoch in range(epochs):
        for i in range(0, len(y_samples), batch_size):
            y_batch = y_samples[i:i + batch_size].to(device)
            x_batch = x_samples[i:i + batch_size].to(device)
            optimizer.zero_grad()
            x_pred = inverse_mlp(y_batch)
            loss = criterion(x_pred, x_batch)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


def compare_outputs_with_inverse_model(model,f_function, secret_encoder, test_loader, secret_dim, inverse_mlp, threshold, n_secret=8, device='cpu'):
    model.eval()
    secret_encoder.eval()
    inverse_mlp.eval()
    total_samples = 0
    total_below_threshold = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sentence_batch = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            for _ in range(n_secret):
                secret_batch1 = torch.randint(0, 2, (sentence_batch.shape[0], secret_dim)).float().to(device)

                output1, _ = model(sentence_batch, attention_mask, secret_batch1)

                embedding_batch1 = secret_encoder(secret_batch1)

                secret_reconstructed = torch.round(inverse_mlp(embedding_batch1))

                output2, _ = model(sentence_batch, attention_mask, secret_reconstructed)

                diff = F.pairwise_distance(output1, output2, p=2)

                total_below_threshold += (diff < threshold).sum().item()
                total_samples += len(diff)

    percentage_below_threshold = total_below_threshold / total_samples * 100
    print(f"Percentage of samples with diff < threshold: {percentage_below_threshold:.4f}%")
    return percentage_below_threshold

# Load config
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test with secret recovery attack")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()

    model_path = args.model_path
    config_path = args.config_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = load_config(config_path)

    num_samples_list = [1000,5000,10000,50000,100000,200000,500000,1000000]
    secret_dim = model_config['model']['secret_dim']
    output_dim = model_config['model']['output_dim']
    threshold = model_config['threshold']
    sentence_model_name = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(sentence_model_name)
    data_collator = DataCollatorWithPadding(tokenizer)

    dataset = TextDataset(args.dataset_path)
    test_size = int(0.1 * len(dataset))
    _, test_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=data_collator,num_workers=1)

    model = load_model(model_config['y_model']['path'], secret_dim, output_dim, sentence_model_name, 768, device)
    secret_encoder, f_function = load_model_and_extract(model_path, model_config, device)
    for num_samples in num_samples_list:
        print(f"Number of Samples: {num_samples}")
        inverse_mlp = InverseMLP(1024, secret_dim).to(device)
        train_inverse_model(secret_encoder, inverse_mlp, secret_dim, epochs=100, num_samples=num_samples, device=device)
        compare_outputs_with_inverse_model(model,f_function, secret_encoder, test_loader, secret_dim, inverse_mlp, threshold, n_secret=8, device=device)
    
