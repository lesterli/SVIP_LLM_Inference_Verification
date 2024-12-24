import torch
from torch.utils.data import DataLoader, random_split
from transformers import DataCollatorWithPadding, AutoTokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import yaml
from models import *
from mydatasets import *
from utils import *
import os

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, outputs, encode_sentences):
        N = len(outputs)

        sentence_loss = 0.0
        for i in range(N):
            sentence_loss += distance_consistency_loss(outputs[i], encode_sentences)
        sentence_loss /= N

        secret_distance = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                secret_distance += distance_margin_loss(outputs[i], outputs[j], self.margin)
        secret_loss = secret_distance / (N * (N - 1) / 2)

        loss = self.alpha * sentence_loss + (1 - self.alpha) * secret_loss
        return loss, sentence_loss, secret_loss

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(model, train_loader, secret_batch_size, secret_dim, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    total_sentence_loss = 0.0
    total_secret_loss = 0.0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        sentence_batch = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = []
        for n in range(secret_batch_size):
            secret_tensor = torch.randint(0, 2, (secret_dim,)).float().to(device)
            secret_batch = secret_tensor.unsqueeze(0).expand(sentence_batch.shape[0], secret_dim)

            output, encode_sentence = model(sentence_batch, attention_mask, secret_batch)
            outputs.append(output)

        loss, sentence_loss, secret_loss = criterion(outputs, encode_sentence)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sentence_loss += sentence_loss.item()
        total_secret_loss += secret_loss.item()

    print(f"Training Loss: {total_loss / len(train_loader)}")
    print(f"Training Sentence Loss: {total_sentence_loss / len(train_loader)}")
    print(f"Training Secret Loss: {total_secret_loss / len(train_loader)}")

def test_model(model, test_loader, secret_batch_size, secret_dim, criterion, device):
    model.eval()
    total_loss = 0.0
    total_sentence_loss = 0.0
    total_secret_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sentence_batch = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = []
            for n in range(secret_batch_size):
                secret_tensor = torch.randint(0, 2, (secret_dim,)).float().to(device)
                secret_batch = secret_tensor.unsqueeze(0).expand(sentence_batch.shape[0], secret_dim)

                output, encode_sentence = model(sentence_batch, attention_mask, secret_batch)
                outputs.append(output)

            loss, sentence_loss, secret_loss = criterion(outputs, encode_sentence)
            total_loss += loss.item()
            total_sentence_loss += sentence_loss.item()
            total_secret_loss += secret_loss.item()

    print(f"Test Loss: {total_loss / len(test_loader)}")
    print(f"Test Sentence Loss: {total_sentence_loss / len(test_loader)}")
    print(f"Test Secret Loss: {total_secret_loss / len(test_loader)}")

def distance_between_two_secrets(model, test_loader, secret_dim, threshold, device, secret_n=8):
    model.eval()
    diffs = []
    over_threshold_count = 0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            sentence_batch = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            for _ in range(secret_n):

                secret_batch1 = torch.randint(0, 2, (sentence_batch.shape[0], secret_dim)).float().to(device)
                secret_batch2 = torch.randint(0, 2, (sentence_batch.shape[0], secret_dim)).float().to(device)


                output1, _ = model(sentence_batch, attention_mask, secret_batch1)
                output2, _ = model(sentence_batch, attention_mask, secret_batch2)

                diff = F.pairwise_distance(output1, output2, p=2)
                diffs.append(diff)

                over_threshold_count += (diff > threshold).sum().item()
                total_count += len(diff)

    diffs_tensor = torch.cat(diffs, dim=0)

    avg_diffs = diffs_tensor.mean().item()
    print(f"Average difference between outputs with different secrets: {avg_diffs}")
    print(f"Percentage of samples with difference greater than {threshold}: {over_threshold_count / total_count * 100:.4f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training labeling model")
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"learnable_y_{config['secret_dim']}_m{config['margin']}")

    tokenizer = AutoTokenizer.from_pretrained(config["sentence_encode_model_name"])
    data_collator = DataCollatorWithPadding(tokenizer)

    dataset = TextDataset(args.dataset_path, tokenizer)
    subset_length = len(dataset) // 10
    dataset = torch.utils.data.Subset(dataset, range(subset_length))

    test_size = int(config["test_split_ratio"] * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=12, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=12, collate_fn=data_collator)

    model = LearnableYModel(secret_dim=config["secret_dim"], output_dim=config["output_dim"], 
                            sentence_model_name=config["sentence_encode_model_name"], sentence_embed_dim=config["sentence_embed_dim"], 
                            output_range_max=config["output_range_max"]).to(device)
    criterion = ContrastiveLoss(margin=config["margin"], alpha=config["alpha"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
    scheduler = None

    # Training loop
    for epoch in range(config["num_epochs"]):
        print(f"Epoch :{epoch}")
        train_model(model, train_loader, config["secret_batch_size"], config["secret_dim"], optimizer, scheduler, criterion, device)
        test_model(model, test_loader, config["secret_batch_size"], config["secret_dim"], criterion, device)
        distance_between_two_secrets(model, test_loader, config["secret_dim"], config["threshold"], device)
        directory = f"./models/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), f'{directory}Ymodel.pth')
