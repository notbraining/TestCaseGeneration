import threading
from tqdm import tqdm
from datasets import load_dataset
import re
import requests
from datasets import Dataset
import json
import asyncio
import random
import aiohttp
import asyncio
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, DeepseekV3ForCausalLM

def main (split: str):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    correct_data = load_from_disk(f"{split}.hf")
    correct = 0
    print(type(correct_data))
    print(correct_data)
    tot = len(correct_data)
    for data in correct_data:
        inputs = tokenizer(correct_data['prompt'], return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        solution_str = re.findall(
                r"<answer>(.*)</answer>", generated, re.DOTALL
        )
        
        if len(solution_str)>0 and solution_str[-1] == correct_data['completions']:
            correct+=1
    print(f"Eval: {correct}/{tot}, Accuracy: {correct/tot}")
    
if __name__ == "__main__":
     main("test")