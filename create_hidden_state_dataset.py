import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from mydatasets import TextDataset
import numpy as np
import os
import h5py
import argparse
from utils import get_model_dimension
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and save hidden states.")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for DataLoader")
    parser.add_argument("--start", type=int, default=0, help="Start of the dataset subset (%)")
    parser.add_argument("--end", type=int, default=10, help="End of the dataset subset (%)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to be used for inference")
    
    return parser.parse_args()

def main():
    args = parse_args()

    seed = 42
    torch.manual_seed(seed)

    print("Start inference and saving hidden states!")

    root_dir = "..."
    save_path = f"{root_dir}/dataset/"
    input_max_length = 48
    model_name = args.model_name
    input_dim = get_model_dimension(model_name)
    print("Hidden dimension:", input_dim)

    if args.multi_gpu:
        model = AutoModel.from_pretrained(model_name, device_map="auto", cache_dir=f"{root_dir}/hf", output_hidden_states=True).half()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).half().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    h5file_path = os.path.join(save_path, f'hidden_states_{args.start}_{args.end}_L{input_max_length}_{model_name.replace("/", "_")}.h5')
    if os.path.exists(h5file_path):
        os.remove(h5file_path)
        print(f"Removed existing file: {h5file_path}")

    train_dataset = TextDataset(f"{root_dir}/text_dataset_train.json")
    subset_length = len(train_dataset) // 100
    train_subset = torch.utils.data.Subset(train_dataset, range(subset_length * args.start, subset_length * args.end))
    print("Dataset index: Start:", subset_length * args.start, "End:", subset_length * args.end)

    def collate_fn(batch, tokenizer, max_length=48):
        texts = [item['text'] for item in batch]
        tokenized_texts = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        return tokenized_texts

    train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer))

    with h5py.File(h5file_path, 'w') as hf:
        hf.create_dataset('hidden_states', shape=(0, input_max_length, input_dim), maxshape=(None, input_max_length, input_dim), dtype=np.float16, compression="gzip")
        hf.create_dataset('input_ids', shape=(0, input_max_length), maxshape=(None, input_max_length), dtype=np.int32, compression="gzip")
        hf.create_dataset('attention_mask', shape=(0, input_max_length), maxshape=(None, input_max_length), dtype=np.int8, compression="gzip")

    def save_hidden_states_and_inputs(hf, last_layer_hidden_states, input_ids, attention_mask):
        batch_size = last_layer_hidden_states.shape[0]

        hf['hidden_states'].resize((hf['hidden_states'].shape[0] + batch_size), axis=0)
        hf['input_ids'].resize((hf['input_ids'].shape[0] + batch_size), axis=0)
        hf['attention_mask'].resize((hf['attention_mask'].shape[0] + batch_size), axis=0)
                               
        hf['hidden_states'][-batch_size:] = last_layer_hidden_states
        hf['input_ids'][-batch_size:] = input_ids
        hf['attention_mask'][-batch_size:] = attention_mask

    model.eval()
    device = model.device

    with h5py.File(h5file_path, 'a') as hf:
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            assert input_ids.shape[1] == input_max_length
            attention_mask = batch['attention_mask'].to(device)
        
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_layer_hidden_states = outputs.hidden_states[-1].detach().cpu().numpy()

            save_hidden_states_and_inputs(hf, last_layer_hidden_states, input_ids.cpu().numpy(), attention_mask.cpu().numpy())

            del input_ids, outputs, last_layer_hidden_states
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
