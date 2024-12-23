import json
import nltk
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the text dataset")
    parser.add_argument("--model_name", type=str,default="openai-community/gpt2-xl")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    return parser.parse_args()

def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def process_data(text, tokenizer, max_length=48,min_length=5):
    sentences = split_into_sentences(text)
    processed_data = []
    for sentence in sentences:
        token_ids = tokenizer.encode(sentence)
        if len(token_ids)>max_length or len(token_ids)<min_length:
            continue

        processed_data.append({
            "texts":sentence
        })
    return processed_data

def main():
    args = parse_args()
    nltk.download('punkt')
    model_name = args.model_name
    dataset_name = args.dataset_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(dataset_name)
    max_examples = 1000  # Limit total examples

    for split in ["train"]:
        processed_data = []
        for example in tqdm(dataset[split]):
            text = example["chosen"] 
            processed_examples = process_data(text,tokenizer)
            processed_data.extend(processed_examples)
            if len(processed_data) >= max_examples:
                break
        
        output_file = f"./datasets/text_dataset_{split}.json"
        with open(output_file, "w") as f:
            json.dump(processed_data, f)

        print(f"Dataset saved as: {output_file}")

if __name__ == "__main__":
    main()
