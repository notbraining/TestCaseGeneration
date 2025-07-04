import threading
from requests.api import get
from tqdm import tqdm
from datasets import load_dataset
import re
import requests
from datasets import Dataset
import json


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


def get_completion(prompt: str) -> str:
    try:
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
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(response.status_code)
        print(response)
        print(response.json)
        return f"{response.json()['choices'][0]['message']['reasoning']}\n{response.json()['choices'][0]['message']['content']}"
    except:
        return get_completion(prompt)


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


def main(split: str, max_length: int):
    dataset = load_dataset("BAAI/TACO", split=split)
    prompts_final = []
    completions_final = []

    for data in tqdm(dataset):
        question = data["question"]
        inputs, outputs = filter_data(data)
        prompts, completions = format_samples(question, inputs, outputs)
        for prompt, completion in zip(prompts, completions):
            model_completion = get_completion(prompt)
            solution_str = re.findall(
                r"<answer>(.*)</answer>", model_completion, re.DOTALL
            )
            if len(solution_str) > 0:
                if solution_str[-1].strip() == completion.strip():
                    prompts_final.append(prompt)
                    completions_final.append(model_completion.strip())
                    if len(prompts_final) >= max_length:
                        save_dataset(
                            prompts_final, completions_final, split, f"{split}.hf"
                        )
                        return
                    continue

    save_dataset(prompts_final, completions_final, split, f"{split}.hf")


if __name__ == "__main__":
    main("test", 10)


# first loop
# takes in TACO dataset 
# return prompt (your system prompt + problem statement + <input><input>) and target output ex "1"

# second loop takes in what first loops return
# return a list of deepseek's answers and MAKES REQUEST IN APRPAPRRA
# i_f deepseskk's answer does NOT match the output; delete it

# last thoing 
# make the dataset with prompt and completion  
# and write to disk
