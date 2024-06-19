import pandas as pd
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# # Log in to Hugging Face
# login(token="your_huggingface_api_token")

# GPT-2 small is the smallest version of the GPT-2 models. 
# It's less resource-intensive compared to larger models but still performs well for many tasks.
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    # logging.debug(f"Tokenization took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    outputs = model.generate(**inputs, max_length=100, do_sample=True)
    # logging.debug(f"Model generation took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # logging.debug(f"Decoding took {time.time() - start_time:.2f} seconds")
    
    return response


def validate_model(dataset_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Initialize counters for validation
    total = len(df)
    correct = 0

    # Loop through the dataset and validate responses
    for index, row in df.iterrows():
        prompt = f"What is the ICD10 code (International Classification of Diseases, Tenth edition) for {row['Intestinal infectious diseases']}?"
        expected_response = row['ICD10_A00-A09']
        
        # Generate model response
        generated_response = generate_response(prompt)
        
        # Validate the response (simple comparison)
        if expected_response.strip().lower() in generated_response.strip().lower():
            correct += 1
        else:
            logging.debug(f"Mismatch for prompt: {prompt}")
            logging.debug(f"Expected: {expected_response}")
            logging.debug(f"Generated: {generated_response}")

    accuracy = correct / total * 100
    logging.info(f"Validation Accuracy: {accuracy:.2f}%")

# Test the function
prompt = "What ICD10 code (International Classification of Diseases, Tenth edition) for tetanus?"
response = generate_response(prompt=prompt)
print(f"Generated response: {response}")

# Validate the model with the dataset
dataset_path = "~/Desktop/cs learning/llms/data/data-verbose.csv"  # Replace with the actual path to your JSON file
validate_model(dataset_path)