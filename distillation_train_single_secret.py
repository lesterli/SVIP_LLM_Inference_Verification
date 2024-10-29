import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import h5py
from models import TransformerGFWithSecret
import tqdm
import wandb
import os
import shutil
import pickle
import yaml

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Distillation Training Script')
    parser.add_argument('--project_name', type=str, default='miv_distillation_single', help='WandB project name')
    parser.add_argument('--file_name_1', type=str, required=True, help='Target model hidden states dataset file name')
    parser.add_argument('--file_name_2', type=str, required=True, help='Alternative model hidden states dataset file name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--test_split_ratio', type=float, default=0.1, help='Test set split ratio')
    parser.add_argument('--verification_model_path', type=str, required=True, help='Path to verification model state dict')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--collected_data_size', type=int, default=1000, help='Size of the collected data under each single secret')
    parser.add_argument('--secret_num', type=int, default=30, help='Number of the collected secrets')
    parser.add_argument('--run_config', type=str, required=True, help='Path to run config')
    return parser.parse_args()

class DistillDataset(Dataset):
    def __init__(self, file_name_1, file_name_2):
        self.hf_1 = file_name_1
        self.hf_2 = file_name_2
        with h5py.File(self.hf_1, 'r') as hf:
            self.length_1 = hf['input_ids'].shape[0]
        with h5py.File(self.hf_2, 'r') as hf:
            self.length_2 = hf['input_ids'].shape[0]
        assert self.length_1 == self.length_2
        self.length = self.length_1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hf_1, 'r') as hf:
            transformer_outputs_1 = hf['hidden_states'][idx]
            attention_mask_1 = hf['attention_mask'][idx]
        with h5py.File(self.hf_2, 'r') as hf:
            transformer_outputs_2 = hf['hidden_states'][idx]
            attention_mask_2 = hf['attention_mask'][idx]
        return {
            'transformer_outputs_1': transformer_outputs_1,
            'attention_mask_1': attention_mask_1,
            'transformer_outputs_2': transformer_outputs_2,
            'attention_mask_2': attention_mask_2
        }

class AdapterMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(AdapterMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
def generate_secret_batch(batch_size, secret_dim, device):
    return torch.randint(0, 2, (batch_size, secret_dim)).float().to(device)


def train(adapter, train_loader, criterion, optimizer, verification_model, device, secret):
    adapter.train()
    total_loss = 0.0

    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()

        transformer_outputs_1 = batch['transformer_outputs_1'].float().to(device)
        transformer_outputs_2 = batch['transformer_outputs_2'].float().to(device)
        attention_mask_1 = batch['attention_mask_1'].to(device)
        attention_mask_2 = batch['attention_mask_2'].to(device)

        transformed_outputs = adapter(transformer_outputs_2)
        secret_batch = secret.expand(transformed_outputs.shape[0], secret.shape[-1])
        transformer_output_1_final = verification_model.forward_no_f(transformer_outputs_1, secret_batch, attention_mask=attention_mask_1)
        transformer_output_2_final = verification_model.forward_no_f(transformed_outputs, secret_batch, attention_mask=attention_mask_2)

        loss = criterion(transformer_output_1_final, transformer_output_2_final)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"train_batch_loss": loss.item()})

    average_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {average_train_loss}")
    return average_train_loss

# Test function
def test(adapter, test_loader, criterion, verification_model, device, secret):
    adapter.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            transformer_outputs_1 = batch['transformer_outputs_1'].float().to(device)
            transformer_outputs_2 = batch['transformer_outputs_2'].float().to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)

            transformed_outputs = adapter(transformer_outputs_2)
            secret_batch = secret.expand(transformed_outputs.shape[0], secret.shape[-1])
            transformer_output_1_final = verification_model.forward_no_f(transformer_outputs_1, secret_batch, attention_mask=attention_mask_1)
            transformer_output_2_final = verification_model.forward_no_f(transformed_outputs, secret_batch, attention_mask=attention_mask_2)

            loss = criterion(transformer_output_1_final, transformer_output_2_final)
            total_test_loss += loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss}")
    return average_test_loss

def extract_model_name(file_name):
    base_name = os.path.basename(file_name)  
    model_name = "_".join(base_name.split('_')[5:7])
    return model_name

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    args = parse_args()
    set_seed(42) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_1 = extract_model_name(args.file_name_1)
    model_name_2 = extract_model_name(args.file_name_2)
    model_name_2 = "gpt2-xl" if model_name_2 == "gpt2-xl_last" else model_name_2
    print(f"-----Distill_{model_name_1}_TO_{model_name_2}-----")

    config = load_config(args.run_config)

    verification_model = TransformerGFWithSecret(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        secret_dim=config['model']['secret_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    verification_model.load_state_dict(torch.load(args.verification_model_path))
    for param in verification_model.parameters():
        param.requires_grad = False

    dataset = DistillDataset(args.file_name_1, args.file_name_2)
    d1 = dataset[0]['transformer_outputs_1'].shape[-1]
    d2 = dataset[0]['transformer_outputs_2'].shape[-1]

    save_dir = f".../distillation_models/{model_name_1}"
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.makedirs(save_dir)

    model_save_dir = f"{save_dir}/{model_name_2}_{args.collected_data_size}"
    if os.path.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir)

    secret_cache = {}

    for secret_idx in range(args.secret_num):
        print(f"-----Secret {secret_idx}-----")

        secret_dim = config['model']['secret_dim']
        secret = generate_secret_batch(1, secret_dim, device)
        secret_cache[secret_idx] = secret.cpu().numpy()
        print("Secret:", secret)
        wandb.init(project=args.project_name, name=f"distill_{model_name_1}_TO_{model_name_2}_secret_{secret_idx}_{args.collected_data_size}_epochs_{args.epochs}")

        # subset_size = int(0.1 * len(dataset))
        sampled_dataset = Subset(dataset, torch.randperm(len(dataset))[:args.collected_data_size])
        assert len(sampled_dataset) == args.collected_data_size

        test_size = int(args.test_split_ratio * len(sampled_dataset))
        train_size = len(sampled_dataset) - test_size

        train_dataset, test_dataset = random_split(sampled_dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        adapter = AdapterMLP(d2, d1).to(device)
        optimizer = optim.Adam(adapter.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(args.epochs):
            print(f"Epoch [{epoch+1}/{args.epochs}]")
            train_loss = train(adapter, train_loader, criterion, optimizer, verification_model, device, secret)
            test_loss = test(adapter, test_loader, criterion, verification_model, device, secret)
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "test_loss": test_loss})

        torch.save(adapter.state_dict(), f"{model_save_dir}/secret_{secret_idx}.pth")    
        wandb.finish()

        del adapter

    secret_cache["d1"] = d1
    secret_cache["d2"] = d2

    save_path = f"{model_save_dir}/secrets.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(secret_cache, f)

if __name__ == "__main__":
    main()


