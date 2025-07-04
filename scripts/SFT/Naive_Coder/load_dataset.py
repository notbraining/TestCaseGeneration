from datasets import load_dataset
import os
from datasets import Dataset
import json
import asyncio
import aiohttp
import multiprocessing
import time
import io
import sys
import builtins
import re
import random
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()


# returns simple test cases for a problem
def filter_data(data: dict, difficulties: list[str]) -> tuple[list[str], list[str]]:
    if data["difficulty"] not in difficulties:
        return ([], [])

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
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"{idx} finished \n")
                        reasoning = data["choices"][0]["message"]["reasoning"]
                        content = data["choices"][0]["message"]["content"]
                        return f"{reasoning}\n{content}"
                    else:
                        print(f"Request failed with status {resp.status}")
            except Exception as e:
                print(f"{idx} failed: {e}\n")
                await asyncio.sleep(1)  # Optional backoff on failure

    return "FAIL"


def format_prompt(question: str) -> str:
    return f"""
**Role:**
You are an expert Python programmer and an expert in writing test cases for competitive programming problems. Your task is to write Python code to solve competitive programming problems.

**Your Responsibilities:**
- Understand the problem and its input constraints.
- Identify and consider potential edge / corner case inputs.
- Reason through how you plan to solve the problem.
- Write out simple solution code in Python for the given problem. The code should:
    - Read input from stdin (e.g., int, str)
    - Output the final answer to stdout
    - Compile and run without errors
    - Determine the correct output for as many cases as possible (without much regard for time complexity)
    - Be relatively simplistic and basic
- Be precise, deterministic, and thorough in your reasoning and solution code.

**Problem Statment:**
{question}
"""


def save_dataset(
    prompts: list[str],
    completions: list[str],
    inputs: list[list[str]],
    outputs: list[list[str]],
    split: str,
    path: str,
):
    print(len(prompts), len(completions), len(inputs), len(outputs))

    # shuffled_list = list(zip(prompts, completions, inputs, outputs))
    # random.shuffle(shuffled_list)
    # prompts, completions, inputs, outputs = zip(*shuffled_list)

    ds = Dataset.from_dict(
        {
            "prompt": prompts,
            "completion": completions,
            "inputs": inputs,
            "output": outputs,
        }
    )

    print("prompts: ", prompts)
    print("completions: ", completions)
    print("inputs: ", inputs)
    print("outputs: ", outputs)

    print("\n\n")
    print(ds)

    print(f"{split} saved to {ds.save_to_disk(f'{split}.hf')}")


def run_code(code: str, test_in: str, output_queue):
    start = time.time()

    if isinstance(test_in, list):
        test_in = " ".join(str(s) for s in test_in)

    # Mock input
    input_lines = iter(test_in.splitlines())
    builtins.input = lambda: next(input_lines)

    # Capture output
    buf = io.StringIO()
    sys.stdout = buf
    try:
        exec(code)
    except Exception as _:
        output_queue.put(("RUNTIME ERROR\n", 0))
        return
    finally:
        sys.stdout = sys.__stdout__

    output_queue.put(
        (
            {
                "output": buf.getvalue(),
                "time": time.time() - start,
            },
            0,
        )
    )


def test_code(
    code: str, inputs: list[str], outputs: list[str], time_limit=5000
) -> bool:
    print("testing code on inputs...")

    for i in range(len(inputs)):
        output_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_code, args=(code, inputs, output_queue))
        p.start()
        p.join(timeout=time_limit)

        if p.is_alive():
            p.terminate()
            print("infinite loop")
            return False
        else:
            try:
                out, _ = output_queue.get(timeout=1)
            except:
                print("failed to collect output")
                return False

        if not (isinstance(out, dict) or out["output"].rstrip() == outputs[i].rstrip()):
            print("incorrect output")
            return False

    print("passed")
    return True


async def main(split: str, difficulties: list[str], max_length: int):
    dataset = load_dataset("BAAI/TACO", split=split)
    prompts_final = []
    completions_final = []
    inputs_final = []
    outputs_final = []

    # creates list of prompts (problem statement + input) and desired completions
    prompts = []
    completions = []
    inputs = []
    outputs = []

    for data in tqdm(dataset):
        question = data["question"]
        test_ins, test_outs = filter_data(data, difficulties)
        prompt = format_prompt(question)

        if len(test_ins) > 0:
            prompts.append(prompt)
            inputs.append(test_ins)
            outputs.append(test_outs)

    prompts = prompts[:max_length]
    inputs = inputs[:max_length]
    outputs = outputs[:max_length]

    print(f"Length of dataset to be given to openrouter: {len(prompts)}")

    tasks = [get_completion(prompts[i], i) for i in range(len(prompts))]
    completions = await asyncio.gather(*tasks)

    print("prompt:")
    print(prompts[0])
    print("completion:")
    print(completions[0])
    print("test cases:")
    print(inputs[0], "\n", outputs[0])

    for i in range(len(prompts)):
        code_str = re.findall(r"```python\n(.*)\n```", completions[i], re.DOTALL)

        if len(code_str) > 0:
            if test_code(code_str[-1].strip(), inputs[i], outputs[i]):
                prompts_final.append(prompts[i])
                completions_final.append(completions[i].strip())
                inputs_final.append(inputs[i])
                outputs_final.append(outputs[i])

                if len(prompts_final) >= max_length:
                    save_dataset(
                        prompts_final,
                        completions_final,
                        inputs_final,
                        outputs_final,
                        split,
                        f"{split}.hf",
                    )
                    break

    # print(completions)
    save_dataset(
        prompts_final,
        completions_final,
        inputs_final,
        outputs_final,
        split,
        f"{split}.hf",
    )


if __name__ == "__main__":
    asyncio.run(main("test", ["EASY"], 100))
