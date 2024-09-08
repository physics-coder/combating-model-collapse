import random
import time
import torch
import json
import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define constants
PROMPT = "People often believe that"
NUM_SAMPLES = 100
MAX_LENGTH = 100

def load_model_and_tokenizer(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

def generate_samples(model, tokenizer, prompt, num_samples, temperature, top_p, top_k):
    inputs = tokenizer(prompt, return_tensors="pt")
    torch.manual_seed(random.randrange(0, 10000))

    start_time = time.time()
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,  # Set do_sample based on temperature
            max_length=MAX_LENGTH,
            use_cache=False,
            num_return_sequences=num_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    end_time = time.time()

    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"Time taken to generate {num_samples} samples: {end_time - start_time:.2f} seconds")
    return decoded_preds

def calculate_distinct_n(samples, n):
    scores = []
    for sample in samples:
        tokens = sample.split()
        if len(tokens) < n:
            scores.append(0)  # Handle cases where sample is shorter than n
            continue
        n_grams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        distinct_n_grams = set(n_grams)
        total_n_grams = len(n_grams)
        
        if total_n_grams == 0:
            scores.append(0)
        else:
            scores.append(len(distinct_n_grams) / total_n_grams)
    
    return sum(scores) / len(scores)  # Return the average score

def run_experiment(model, tokenizer, config):
    samples = generate_samples(model, tokenizer, PROMPT, NUM_SAMPLES, 
                               config['temperature'], config['top_p'], config['top_k'])
    
    results = {}
    for n in [1, 2, 3]:
        distinct_n = calculate_distinct_n(samples, n)
        results[f'distinct-{n}'] = distinct_n
    
    # Add average sample length to results
    avg_length = sum(len(sample.split()) for sample in samples) / len(samples)
    results['avg_sample_length'] = avg_length
    
    # Add the first sample as an example
    results['example_sample'] = samples[0]
    
    return results, samples[0]

def main(model_path):
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    experiments = {
        'baseline': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0},
        'nucleus_sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 0},
        'conservative_sampling': {'temperature': 0.7, 'top_p': 0.9, 'top_k': 0},
        'top_k_sampling': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
    }
    
    results = {}
    example_samples = {}
    for name, config in experiments.items():
        print(f"Running {name} experiment...")
        results[name], example_samples[name] = run_experiment(model, tokenizer, config)
    
    # Save diversity results to JSON file
    diversity_output_file = os.path.join(model_path, 'diversity_results.json')
    with open(diversity_output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save example samples to a separate JSON file
    samples_output_file = os.path.join(model_path, 'example_samples.json')
    with open(samples_output_file, 'w') as f:
        json.dump(example_samples, f, indent=2)
    
    print(f"Diversity results saved to {diversity_output_file}")
    print(f"Example samples saved to {samples_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-2 sampling experiments")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    args = parser.parse_args()
    
    main(args.model_path)