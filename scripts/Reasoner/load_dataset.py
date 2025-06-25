import argparse
import re
import os
from typing import cast
import json
import datasets

from datasets.load import Dataset, DatasetDict
from verl.utils.hdfs_io import copy, makedirs
# added this header


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "BAAI/TACO"

    dataset: DatasetDict = cast(DatasetDict, datasets.load_dataset(data_source, "ALL"))

    train_dataset: Dataset = dataset["train"]
    test_dataset: Dataset = dataset["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw

            test_cases = example.pop("input_output")
            try:
                test_cases = json.loads(test_cases)
            except:
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {"inputs": [], "outputs": []},
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "question": question_raw,
                    },
                }
                return data

            inputs: list[str] = test_cases["inputs"]
            outputs: list[str] = test_cases["inputs"]


            def isValid(test_in: str, test_out: str):
                test_in = test_in.strip()
                test_out = test_out.strip()

                if len(test_in) > 20 or len(test_out) > 10:
                    return False

                for num in test_in.split():
                    if not num.isdigit():
                        continue
                    if int(num) > 25:
                        return False

                for num in test_out.split():
                    if not num.isdigit():
                        continue
                    if int(num) > 25:
                        return False
                return True

            def remove_long(data) -> bool:
                if isinstance(data[0], str) and isinstance(data[1], str):
                    return isValid(data[0], data[1])
                return False

            sys_prompt = f"""
You will be given a competitive programming problem with small input test cases. 
You need to find the correct output for the given input by reasoning step by step through the test cases without writing any code.
Place your final answer between <answer> and </answer>.
"""

            try:
                inputs, outputs = map(
                    list, zip(*list(filter(remove_long, zip(inputs, outputs))))
                )
                data = {
                    "data_source": data_source,
                    "prompt": [
                         {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {"inputs": inputs, "outputs": outputs},

                    },
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "question": question_raw,
                    },
                }
                return data
            except:
                #print(idx)
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {"inputs": [], "outputs": []},
                    },
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "question": question_raw,
                    },
                }
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "TACO_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "TACO_test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
