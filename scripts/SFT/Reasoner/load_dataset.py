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

# returns simple test cases for a problem
def filter_data(data: dict) -> tuple[list[str], list[str]]:
    input_output = json.loads(data["input_output"])
    inputs = input_output["inputs"]
    outputs = input_output["outputs"]
    valid_in = []
    valid_out = []

    for sample_in, sample_out in zip(inputs, outputs):
        case_valid = False

        # checks if sample inputs and outputs are strings
        if isinstance(sample_in, str) and isinstance(sample_out, str):
            case_valid = True

            # prunes lengthy test cases
            if len(sample_in) > 60 or len(sample_out) > 37:
                # print("input / output too long")
                case_valid = False
                continue

            # checks for large numbers
            for num in sample_in.strip().split():
                if len(num) > 10 or (num.isdigit() and int(num) > 30):
                    # print("input number too large / word too long")
                    case_valid = False
                    break

            for num in sample_out.strip().split():
                if len(num) > 10 or (num.isdigit() and int(num) > 30):
                    # print("output number too large / word too long")
                    case_valid = False
                    break

        if case_valid:
            # print("valid test case")
            valid_in.append(sample_in)
            valid_out.append(sample_out)
        # print()

    return (valid_in, valid_out)


async def get_completion(prompt: str, idx: int) -> str:
    print(f"Making request with index: {idx}")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {
            "exclude": False,
            "enabled": True,
        },
    }

    async with aiohttp.ClientSession() as session:
        for i in range(5):  # Retry loop
            try:
                async with session.post(url, headers=headers, json=payload,timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"{idx} finished \n")
                        reasoning = data['choices'][0]['message']['reasoning']
                        content = data['choices'][0]['message']['content']
                        return f"{reasoning}\n{content}"
                    else:
                        print(f"Request failed with status {resp.status}")
            except Exception as e:
                print(f"fail: {e}\n")
                await asyncio.sleep(1)  # Optional backoff on failure

    return "None"

def format_samples(
    question: str, inputs: list[str], outputs: list[str]
) -> tuple[list[str], list[str]]:
    prompts = []
    completions = []
    for input, output in zip(inputs, outputs):
        prompts.append(format_prompt(question, input))
        completions.append(f"{output}")
    return prompts, completions


def format_prompt(question: str, input: str) -> str:
    return f"Without writing code, read through this competitive programming question and find the answer for this given input at the end, and put your final answer between 2 tags <answer> and </answer>. The question is given between <question> and </question>, find the output of the input given in <input> and </input>. <question>\n{question}\n</question>\n\n<input>\n{input}</input>. Also, remember to not write code, and just find the output with the given input in the tags."


def save_dataset(prompts: list[str], completions: list[str], split: str, path: str):
    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})
    print(f"{split} saved to {ds.save_to_disk(f'{split}.hf')}")


async def main(split: str, max_length: int):
    dataset = load_dataset("BAAI/TACO", split=split)
    prompts_final = []
    completions_final = []
    
    # creates list of prompts (problem statement + input) and desired completions
    prompts = []
    completions = []

    for data in tqdm(dataset):
        questions = data["question"]
        inputs, outputs = filter_data(data)
        prompt, completion = format_samples(questions, inputs, outputs)
        prompts.extend(prompt)
        completions.extend(completion)
    prompts = prompts[:max_length]
    completions = prompts[:max_length]

    print(f"Length of dataset to be given to openrouter: {len(prompts)}")

    tasks = [get_completion(prompts[i], i) for i in range(len(prompts))]
    model_completions = await asyncio.gather(*tasks)
    print("Results:", model_completions)

    for i in range (len(prompts)):
        solution_str = re.findall(
                r"<answer>(.*)</answer>", model_completions[i], re.DOTALL
            )
        
        if len(solution_str) > 0:
            if solution_str[-1].strip() == completions[i].strip():
                prompts_final.append(prompts[i])
                completions_final.append(model_completions[i].strip())
                if len(prompts_final) >= max_length:
                    save_dataset(
                        prompts_final, completions_final, split, f"{split}.hf"
                    )
                    break

    
    # print(completions)
    # shuffles dataset to split up test cases of the same problem
    shuffled_list = list(zip(prompts_final, completions_final))
    random.shuffle(shuffled_list)
    prompts_final, completions_final = zip(*shuffled_list)
    save_dataset(prompts_final, completions_final, split,  f"{split}.hf")

        

if __name__ == "__main__":
    asyncio.run(main("test", 100))



