from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer #AutoModelForMaskedLM for Albert
from huggingface_hub import login
import logging
import time

# Log in to Hugging Face
# login(token="your_huggingface_api_token")
logging.basicConfig(level=logging.DEBUG)

# Load the model and tokenizer
# ALBERT is designed to be more parameter-efficient and faster compared to BERT
# model_name = "albert/albert-base-v2" #"meta-llama/Llama-2-7b-hf"

# GPT-2 small is the smallest version of the GPT-2 models. 
# It's less resource-intensive compared to larger models but still performs well for many tasks.
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)

# def generate_response(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     print("inputs worked")
#     outputs = model.generate(**inputs, max_length=100, do_sample=True)
#     print("outputs worked")
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("response worked")
#     return response

def generate_response(prompt):
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    logging.debug(f"Tokenization took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    outputs = model.generate(**inputs, max_length=100, do_sample=True)
    logging.debug(f"Model generation took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.debug(f"Decoding took {time.time() - start_time:.2f} seconds")
    
    return response


# prompt = "What is the ICD-10 code for diabetes?"
# response = generate_response(prompt=prompt)
# print(response)

# Create Flask app
app = Flask(__name__)

import threading

def background_task(prompt, result):
    response = generate_response(prompt)
    result['response'] = response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    result = {}
    thread = threading.Thread(target=background_task, args=(prompt, result))
    thread.start()
    thread.join()  # This will block until the background task is done
    return jsonify(result)

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     logging.debug(f"Received request data: {data}")
#     prompt = data.get('prompt', '')
#     logging.debug(f"Prompt: {prompt}")
#     response = generate_response(prompt)
#     logging.debug(f"Generated response: {response}")
#     return jsonify({'response': response})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'response': 'pong'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)