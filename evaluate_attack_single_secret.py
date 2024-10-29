import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaTokenizer, get_linear_schedule_with_warmup, AutoTokenizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models import *
from mydatasets import *
from utils import *
from sklearn.metrics import precision_score, recall_score
import pickle 
import argparse
import os
from distillation_train_single_secret import AdapterMLP
import json
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def transform_tokenize(input_ids,origin_tokenizer,new_tokenizer):
    decoded_text = origin_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return new_tokenizer(decoded_text, return_tensors='pt', padding=True)

def generate_secret_batch(batch_size, secret_dim, device):
    return torch.randint(0, 2, (batch_size, secret_dim)).float().to(device)

def evaluate_pipeline(dataloader, verification_model, llm_tokenizer, y_model_tokenizer, y_model, device, secret, 
                          alter_model_name=None, adapter=None, d1=None, d2=None):
    verification_model.eval()
    all_diffs_adapter = []  
    all_diffs_random = []  

    with torch.no_grad():
        if alter_model_name is not None:
            alter_model = AutoModel.from_pretrained(alter_model_name, output_hidden_states=True).half().to(device)
            alter_tokenizer = AutoTokenizer.from_pretrained(alter_model_name)
            alter_tokenizer.pad_token = alter_tokenizer.eos_token

        for batch in tqdm(dataloader):
            attention_mask = batch['attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            transformer_outputs = batch['transformer_outputs'].to(device).float()
            # Decode input_ids using LlamaTokenizer and encode using y_model_tokenizer
            encoded_inputs = transform_tokenize(input_ids,llm_tokenizer, y_model_tokenizer).to(device)

            assert adapter is not None
            assert alter_model_name is not None

            alter_encoded_inputs = transform_tokenize(input_ids,llm_tokenizer,alter_tokenizer).to(device)
            with torch.no_grad():
                alter_model_outputs = alter_model(input_ids=alter_encoded_inputs['input_ids'], attention_mask=alter_encoded_inputs['attention_mask'])
            alter_model_last_layer_hidden_states = alter_model_outputs.hidden_states[-1].float()
            adapted_attention_mask = alter_encoded_inputs['attention_mask']

            secret_batch = secret.expand(transformer_outputs.shape[0], secret.shape[-1])
            y_model_outputs, _ = y_model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], secret_batch)
            #adapter
            adapted_hidden_states_adapter = adapter(alter_model_last_layer_hidden_states)
            #random
            projection_matrix = torch.randn(d2, d1).to(device)
            projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)
            adapted_hidden_states_random = torch.matmul(alter_model_last_layer_hidden_states, projection_matrix)
            
            verification_model_outputs_adapter = verification_model(adapted_hidden_states_adapter, secret_batch, attention_mask=adapted_attention_mask)
            verification_model_outputs_random = verification_model(adapted_hidden_states_random, secret_batch, attention_mask=adapted_attention_mask)

            diff_random = F.pairwise_distance(verification_model_outputs_random, y_model_outputs, p=2)
            diff_adapter = F.pairwise_distance(verification_model_outputs_adapter, y_model_outputs, p=2)

            all_diffs_random.extend(diff_random.cpu().numpy().tolist())
            all_diffs_adapter.extend(diff_adapter.cpu().numpy().tolist())

    return np.array(all_diffs_random), np.array(all_diffs_adapter)

def main():
    parser = argparse.ArgumentParser(description="Evaluate attack under single secret case")
    seed = 42
    parser.add_argument('--verification_model_path', type=str, required=True, help="Path to the verification model")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for dataloader")
    parser.add_argument('--threshold', type=float, default=10, help="Threshold for the difference metric")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the dataset file")
    parser.add_argument('--distillation_models_path', type=str, required=True, help="Path to the distillation models")
    parser.add_argument('--run_config', type=str, required=True, help='Path to run config')

    args = parser.parse_args()
    config = load_config(args.run_config)
    test_split_ratio = 1.00

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load some configs
    secrets_file_path = os.path.join(args.distillation_models_path, "secrets.pkl")
    secret_num = len(os.listdir(args.distillation_models_path)) - 1
    with open(secrets_file_path, 'rb') as f:
        secret_cache = pickle.load(f)
    d1 = secret_cache["d1"]
    d2 = secret_cache["d2"]
    model_name = args.distillation_models_path.split("/")[-2].replace("_","/")
    alter_model_name_and_collected_data_size = args.distillation_models_path.split("/")[-1]
    alter_model_name = alter_model_name_and_collected_data_size.rsplit("_",1)[0].replace("_","/")
    collected_data_size = alter_model_name_and_collected_data_size.rsplit("_",1)[1]

    print("Model name:", model_name, "Alter_model_name:", alter_model_name)

    # Load tokenizer and model
    llm_tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    y_model_tokenizer = AutoTokenizer.from_pretrained(config['y_model']['sentence_encode_model_name'])
    y_model = LearnableYModel(secret_dim=config['model']['secret_dim'], output_dim=config['model']['output_dim'],
                              sentence_model_name=config['y_model']['sentence_encode_model_name'],
                              sentence_embed_dim=config['y_model']['sentence_encode_dim']).to(device)
    y_model.load_state_dict(torch.load(config['y_model']['path']))

    _, test_loader = create_dataloaders(args.data_file, None, args.batch_size, test_split_ratio, num_workers=8)
    input_dim = get_model_dimension(model_name)
    verification_model = TransformerGFWithSecret(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        secret_dim=config['model']['secret_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    verification_model.load_state_dict(torch.load(args.verification_model_path))

    y_model.eval()
    verification_model.eval()
    
    results = {}

    random_asr = []
    adapter_asr = []

    for secret_idx in range(secret_num):
        print(f"Evaluating secret {secret_idx}...")
        secret = torch.tensor(secret_cache[secret_idx]).to(device)
        adapter_path = os.path.join(args.distillation_models_path, f"secret_{secret_idx}.pth")
        adapter = AdapterMLP(d2, d1).to(device)
        adapter.load_state_dict(torch.load(adapter_path))

        all_diffs_random, all_diffs_adapter = evaluate_pipeline(test_loader, verification_model, 
                                                                llm_tokenizer, y_model_tokenizer, y_model, 
                                                                device, secret, alter_model_name=alter_model_name, 
                                                                adapter=adapter, d1=d1, d2=d2)
        
        print(f"Length of all_diffs: {len(all_diffs_random)}")
        below_threshold_ratio_random = np.sum(all_diffs_random < args.threshold) / len(all_diffs_random)
        below_threshold_ratio_adapter = np.sum(all_diffs_adapter < args.threshold) / len(all_diffs_adapter)

        print(f"Random: Percentage of samples with L2 below {args.threshold}: {below_threshold_ratio_random:.2f}")
        print(f"Adapter: Percentage of samples with L2 below {args.threshold}: {below_threshold_ratio_adapter:.2f}")

        results[f"{secret_idx}_random"] = (below_threshold_ratio_random, all_diffs_random.tolist())
        results[f"{secret_idx}_adapter"] = (below_threshold_ratio_adapter, all_diffs_adapter.tolist())

        model_name_save = model_name.replace("/", "_")
        alter_model_save = alter_model_name.replace("/", "_")
        random_asr.append(below_threshold_ratio_random)
        adapter_asr.append(below_threshold_ratio_adapter)
        save_path = f".../results_{model_name_save}_{alter_model_save}_{collected_data_size}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f)

    results["average_random"] = np.mean(np.array(random_asr))
    results["average_adapter"] = np.mean(np.array(adapter_asr))
    with open(save_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()