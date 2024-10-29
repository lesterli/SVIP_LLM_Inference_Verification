import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from models import *
from mydatasets import *
from utils import *
import pickle 
import argparse
import yaml
import os

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

def evaluate_pipeline(dataloader, verification_model, llm_tokenizer, y_model_tokenizer, y_model, device, secret_caches,
                          model_name=None, alter_model_name=None, random_hidden_states=False, secret_batch_size=20):
    verification_model.eval()
    all_diffs = []  

    with torch.no_grad():
        if alter_model_name is not None and alter_model_name != model_name:
                alter_model = AutoModel.from_pretrained(alter_model_name, output_hidden_states=True).half().to(device)
                alter_tokenizer = AutoTokenizer.from_pretrained(alter_model_name)
                alter_tokenizer.pad_token = alter_tokenizer.eos_token

        for batch in tqdm(dataloader):
            attention_mask = batch['attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            transformer_outputs = batch['transformer_outputs'].to(device).float()
            encoded_inputs = transform_tokenize(input_ids,llm_tokenizer, y_model_tokenizer).to(device)

            if alter_model_name is not None:
                if alter_model_name != model_name:
                    alter_encoded_inputs = transform_tokenize(input_ids,llm_tokenizer,alter_tokenizer).to(device)
                    with torch.no_grad():
                        alter_model_outputs = alter_model(input_ids=alter_encoded_inputs['input_ids'], attention_mask=alter_encoded_inputs['attention_mask'])
                    
                    alter_model_last_layer_hidden_states = alter_model_outputs.hidden_states[-1].float()
                    alter_dim = alter_model_last_layer_hidden_states.shape[-1]
                    target_dim = transformer_outputs.shape[-1]
                    projection_matrix = torch.randn(alter_dim, target_dim).to(device)
                    projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)
                    projected_hidden_states = torch.matmul(alter_model_last_layer_hidden_states, projection_matrix)
                    projected_attention_mask = alter_encoded_inputs['attention_mask']
                else:
                    projected_hidden_states = transformer_outputs
                    projected_attention_mask = attention_mask

            batch_diffs = []
            for secret_idx in range(secret_batch_size):
                secret = secret_caches[secret_idx:secret_idx+1, :]
                secret_batch = secret.expand(input_ids.shape[0], secret.shape[-1])
                with torch.no_grad():
                    y_model_outputs, _ = y_model(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], secret_batch)

                if alter_model_name is not None:
                    verification_model_outputs = verification_model(projected_hidden_states, secret_batch, attention_mask=projected_attention_mask)
                elif random_hidden_states:
                    verification_model_outputs = verification_model(torch.randn_like(transformer_outputs).to(device), secret_batch, attention_mask=attention_mask)
                else:
                    raise

                # Calculate distance between model_outputs and y_model_outputs
                diff = F.pairwise_distance(verification_model_outputs, y_model_outputs, p=2)
                batch_diffs.append(diff)
            
            batch_diffs = torch.stack(batch_diffs, dim=0)
            all_diffs.extend(batch_diffs.mean(dim=0).cpu().numpy().tolist())

    return all_diffs

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline performance")
    seed = 42
    parser.add_argument('--model_name', type=str, required=True, help="Name of the specified LLM model")
    parser.add_argument('--verification_model_path', type=str, required=True, help="Path to the verification model")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for dataloader")
    parser.add_argument('--secret_batch_size', type=int, default=20, help="Secret batch size")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the dataset file")
    parser.add_argument('--run_config', type=str, required=True, help='Path to run config')

    args = parser.parse_args()
    config = load_config(args.run_config)
    test_split_ratio = 1.00

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    llm_tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    y_model_tokenizer = AutoTokenizer.from_pretrained(config['y_model']['sentence_encode_model_name'])
    y_model = LearnableYModel(secret_dim=config['model']['secret_dim'], output_dim=config['model']['output_dim'],
                              sentence_model_name=config['y_model']['sentence_encode_model_name'],
                              sentence_embed_dim=config['y_model']['sentence_encode_dim']).to(device)
    y_model.load_state_dict(torch.load(config['y_model']['path']))

    _, test_loader = create_dataloaders(args.data_file, None, args.batch_size, test_split_ratio)
    input_dim = get_model_dimension(args.model_name)
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
    alter_model_names = [
        "gpt2-xl",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-j-6B",
        "facebook/opt-6.7b",
        "lmsys/vicuna-7b-v1.5",
        "meta-llama/Llama-2-7b-hf"
    ]
    alter_model_names.append(args.model_name)

    secret_caches = generate_secret_batch(args.secret_batch_size, config['model']['secret_dim'], device)

    for alter_model_name in alter_model_names:
        print(f"Evaluating {alter_model_name}...")
        all_diffs = evaluate_pipeline(test_loader, verification_model, llm_tokenizer, y_model_tokenizer, y_model, 
                      device, secret_caches, model_name=args.model_name,alter_model_name=alter_model_name,
                      secret_batch_size=args.secret_batch_size, random_hidden_states=False)
        results[alter_model_name.split('/')[-1]] = np.array(all_diffs)

    print("Evaluating Random...")
    all_diffs = evaluate_pipeline(test_loader, verification_model, llm_tokenizer, y_model_tokenizer, y_model, 
                    device, secret_caches, model_name=args.model_name,alter_model_name=None,
                    secret_batch_size=args.secret_batch_size, random_hidden_states=True)
    results["random"] = np.array(all_diffs)

    model_name_save = args.model_name.split('/')[-1]
    save_path = f"./pipeline_evaluation_results/results_{model_name_save}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()