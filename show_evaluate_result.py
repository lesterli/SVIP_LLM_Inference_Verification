import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

def plot_evaluation_result(evaluate_result_path, threshold):
    with open(evaluate_result_path, 'rb') as file:
        data = pickle.load(file)

    diffs_data = []
    below_threshold_ratios = []
    keys = list(data.keys())
    
    if len(keys) > 6:
        keys.insert(0, keys.pop(6))

    for label in keys:
        diffs = data.get(label)
        diffs_data.append(diffs)
        below_threshold_ratio = np.mean(diffs < threshold) * 100
        below_threshold_ratios.append(below_threshold_ratio)

    print("Below Threshold Ratios:")
    for i, label in enumerate(keys):
        print(f"{label}: {below_threshold_ratios[i]:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'cyan', 'green', 'magenta', 'purple', 'red', 'yellow', 'brown']
    alt_model_name = {
        'gpt2-xl': 'GPT2-XL', 'gpt-neo-2.7B': 'GPT-NEO-2.7B', 'gpt-j-6B': 'GPT-J-6B', 
        'opt-6.7b': 'OPT-6.7B', 'vicuna-7b-v1.5': 'Vicuna-7B', 'Llama-2-7b-hf': 'Llama-2-7B', 
        'random': 'Random'
    }
    spec_model_name = {
        'Llama-2-13b-hf': 'Llama-2-13B', 'gpt-neox-20b': 'GPT-NeoX-20B', 'opt-30b': 'OPT-30B',
        'falcon-40b': 'Falcon-40B', 'Meta-Llama-3.1-70B': 'Llama-3.1-70B'
    }

    for j, diffs in enumerate(diffs_data):
        label_key = keys[j]
        plot_label = alt_model_name.get(label_key, label_key) if label_key not in spec_model_name else spec_model_name[label_key]
        weights = np.ones_like(diffs) / len(diffs) * 100
        ax.hist(diffs, bins=100, alpha=1.0 if j == 0 else 0.5, color=colors[j], label=plot_label, weights=weights)

    title = spec_model_name.get(keys[0], keys[0])
    ax.set_title(f"Specified Model: {title}", fontsize=20)
    ax.set_xlabel("L2 Distance Between Proxy Task Output and Label", fontsize=20)
    ax.set_ylabel("Frequency (%)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2)
    ax.legend(loc="upper right", fontsize=16, frameon=False)
    ax.grid(False)

    plt.savefig('./results/evaluation_result.png', format='png')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Evaluation Results')
    parser.add_argument('--evaluate_result_path', type=str, required=True, help='Path to the evaluation result .pkl file')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for L2 distance')
    
    args = parser.parse_args()
    plot_evaluation_result(args.evaluate_result_path, args.threshold)
