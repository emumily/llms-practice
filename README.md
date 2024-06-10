Practicing transformers and llms, specificially [openai-community/gpt2](https://huggingface.co/openai-community/gpt2).

Originally, I was going to use meta-llama/Llama-2-7b-hf but ran into some issues with size constraints of the model and limited computing power. 

I also utilized the Flask app to make it similar to a chatbot. Please note this was done in about 2-3 days and I have not bothered to add other features.

Prompts I gave were "What is the ICD 10 code for diabetes?"

To run:
```python3 llama-test.py```

Then open another terminal:
```curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"prompt": "What is the ICD 10 code for diabetes?"}' ```