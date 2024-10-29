import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoTokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import yaml
import argparse
from models import *
from mydatasets import *
from utils import *
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def transform_tokenize(input_ids, origin_tokenizer, new_tokenizer):
    decoded_text = origin_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return new_tokenizer(decoded_text, return_tensors='pt', padding=True, truncation=True)

def generate_secret_batch(batch_size, secret_dim, device):
    return torch.randint(0, 2, (batch_size, secret_dim)).float().to(device)

def train_model(dataloader, model, llm_tokenizer, y_model_tokenizer, y_model, criterion, optimizer, scheduler, num_epoch, device, secret_dim, secret_batch_size=4,contrastive_loss_weight=0.5,margin=5.0):
    model.train()
    total_loss = 0.0
    total_train_loss = 0.0
    total_secret_loss = 0.0   
    for batch in tqdm(dataloader):
        train_loss = 0.0
        secret_loss = 0.0
        attention_mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        transformer_outputs = batch['transformer_outputs'].to(device).float()

        encoded_inputs = transform_tokenize(input_ids, llm_tokenizer, y_model_tokenizer).to(device)

        for _ in range(secret_batch_size):
            secret_batch = generate_secret_batch(input_ids.shape[0], secret_dim, device)
            with torch.no_grad():
                y_model_outputs, _ = y_model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], secret_batch)

            model_outputs,secret_encoded = model.forward_train(transformer_outputs, secret_batch, attention_mask=attention_mask)

            train_loss += criterion(model_outputs, y_model_outputs)
            secret_loss += contrastive_batch_loss(secret_encoded,margin=margin)
            
        loss = secret_loss * contrastive_loss_weight + train_loss * (1-contrastive_loss_weight)

        loss /= secret_batch_size
        total_loss += loss.item()
        total_train_loss += (train_loss.item() / secret_batch_size)
        total_secret_loss += (secret_loss.item() / secret_batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    avg_loss = total_loss / len(dataloader)
    avg_train_loss = total_train_loss / len(dataloader)
    avg_secret_loss = total_secret_loss / len(dataloader)
    print(f'Epoch {num_epoch}, Loss: {avg_loss}, Regression Loss: {avg_train_loss}, Secret Loss: {avg_secret_loss}')

def evaluate_model(dataloader, model, llm_tokenizer, y_model_tokenizer, y_model, criterion, device, secret_dim, threshold=0.01, secret_mistach=False, secret_batch_size=8):
    model.eval()
    test_loss = 0.0
    total_samples = 0
    below_threshold_count = 0
    total_diff = 0
    all_diffs = [] 

    with torch.no_grad():
        for batch in tqdm(dataloader):
            loss = 0.0
            attention_mask = batch['attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            transformer_outputs = batch['transformer_outputs'].to(device).float()
            encoded_inputs = transform_tokenize(input_ids, llm_tokenizer, y_model_tokenizer).to(device)
            for _ in range(secret_batch_size):
                secret_batch = generate_secret_batch(input_ids.shape[0], secret_dim, device)
                with torch.no_grad():
                    y_model_outputs, _ = y_model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], secret_batch)

                if secret_mistach:
                    secret_batch_diff = generate_secret_batch(input_ids.shape[0], secret_dim, device)
                    model_outputs = model(transformer_outputs, secret_batch_diff, attention_mask=attention_mask)
                else:
                    model_outputs = model(transformer_outputs, secret_batch, attention_mask=attention_mask)

                loss += criterion(model_outputs, y_model_outputs)

                diff = torch.nn.functional.pairwise_distance(model_outputs, y_model_outputs, p=2)

                all_diffs.extend(diff.cpu().numpy())

                below_threshold_count += (diff < threshold).sum().item()
                total_samples += diff.size(0)
                total_diff += diff.sum().item()

            loss /= secret_batch_size
            test_loss += loss.item()

    avg_loss = test_loss / len(dataloader)
    below_threshold_ratio = below_threshold_count / total_samples * 100
    avg_diff = total_diff / total_samples

    all_diffs = np.array(all_diffs)
    sorted_diffs = np.sort(all_diffs)
    threshold_95_percentile = np.percentile(sorted_diffs, 95)

    print(f"Average difference (L2 distance): {avg_diff:.4f}")
    print(f"95th percentile threshold: {threshold_95_percentile:.4f}")
    return avg_loss, below_threshold_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training proxy task model")
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config_path)

    torch.manual_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    llm_tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    y_model_tokenizer = AutoTokenizer.from_pretrained(config['y_model']['sentence_encode_model_name'])
    y_model = LearnableYModel(secret_dim=config['model']['secret_dim'], output_dim=config['model']['output_dim'],
                              sentence_model_name=config['y_model']['sentence_encode_model_name'],
                              sentence_embed_dim=config['y_model']['sentence_encode_dim'],
                              output_range_max=config['y_model']['output_range_max']).to(device)
    y_model.load_state_dict(torch.load(config['y_model']['path']))
    y_model.eval()

    model = TransformerGFWithSecret(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        secret_dim=config['model']['secret_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)


    train_loader, test_loader = create_dataloaders(config['dataset_path'], None, config['batch_size'],
                                                   config['test_split_ratio'], num_workers=config['num_workers'])


    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'],
                                                num_training_steps=len(train_loader) * config['num_epochs'])

    for epoch in range(config['num_epochs']):
        train_model(train_loader, model, llm_tokenizer, y_model_tokenizer, y_model, criterion, optimizer, scheduler, epoch, device, config['model']['secret_dim'], config['secret_batch_size'],config['contrastive_loss_weight'],config['margin'])
        avg_loss, below_threshold_ratio = evaluate_model(test_loader, model, llm_tokenizer, y_model_tokenizer, y_model, criterion, device, config['model']['secret_dim'], threshold=config['threshold'])
        print(f'Epoch {epoch}, Validation Loss: {avg_loss}, Samples below threshold {config["threshold"]}: {below_threshold_ratio:.4f}%')
        directory = f"./models/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), f'{directory}proxy_model_{config['model']['name'].split('/')[-1]}.pth')
