from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import numpy as np
import json
import h5py
import time
import torch

class TokenPredictionDataset(Dataset):
    def __init__(self, dataset_path, token_to_index):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.input_ids = [item['tokens'] for item in data]
        self.token_to_index = token_to_index
        self.length = len(self.input_ids)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        labels = np.zeros(len(self.token_to_index), dtype=np.float32)
        for token in input_ids:
            if token in self.token_to_index:
                labels[self.token_to_index[token]] = 1
        return {"input_ids": input_ids,"labels":labels}

class TextDataset(Dataset):
    def __init__(self, file_path, max_length_before_padding=48):
        self.max_length = max_length_before_padding
        with open(file_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"text": item["texts"]}


class PrecomputedHiddenStatesDataset(Dataset):
    def __init__(self, h5file_path, token_to_index=None):
        with h5py.File(h5file_path, 'r') as hf:
            self.input_ids = hf['input_ids'][:]
            self.transformer_outputs = hf['hidden_states'][:]
            self.attention_mask = hf['attention_mask'][:]
        self.token_to_index = token_to_index
        self.length = self.input_ids.shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        transformer_output = self.transformer_outputs[idx]
        attention_mask = self.attention_mask[idx]

        if not self.token_to_index:
            return {
                "transformer_outputs": transformer_output,
                "attention_mask": attention_mask,
                "input_ids": input_ids
            }
            

def create_token_mapping(tokens_to_test):
    return {token: idx for idx, token in enumerate(tokens_to_test)}

def calculate_class_weights(token_counts, tokens_to_test, total_counts):
    class_weights = np.zeros(len(tokens_to_test))

    for token_id in tokens_to_test:
        if str(token_id) in token_counts:
            class_weights[tokens_to_test.index(token_id)] = ((1 - token_counts[str(token_id)] / total_counts) - 0.5) * 0.8 + 0.5

    return class_weights

def create_dataloaders(h5file_path, token_to_index, batch_size, test_split_ratio,num_workers=64,last_token_dataset=True):
    start = time.time()


    dataset = PrecomputedHiddenStatesDataset(h5file_path, token_to_index)
    print("Finish Loading Dataset in ",time.time()-start)
    test_size = int(test_split_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(len(train_dataset),len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    return train_loader, test_loader
