import argparse
import re
import os
from typing import cast
import json
import datasets
import pandas as pd
from datasets.load import Dataset, DatasetDict
from verl.utils.hdfs_io import copy, makedirs
from tqdm import tqdm

if __name__ == "__main__":
    data_source = "BAAI/TACO"
    dataset: DatasetDict = cast(DatasetDict, datasets.load_dataset(data_source, "ALL"))
    train_dataset: Dataset = dataset["train"]
    test_dataset: Dataset = dataset["test"]

    def make_parquet(split):
        def make_format(question: str, inputs: list[str], outputs: list[str]):
            prompts = []
            ground_truth = []
            for input, output in zip(inputs, outputs):
                if isinstance(input, str) and isinstance(output, str):
                    query = question + "\n" + input
                    answer = output
                    prompts.append(query)
                    ground_truth.append(answer)
            return prompts, ground_truth

        prompts = []
        ground_truth = []
        print("formatting prompts + groundtruth...")
        for sample in tqdm(dataset[split]):
            try:
                question = sample["question"]
                inputs = json.loads(sample["input_output"])["inputs"]
                outputs = json.loads(sample["input_output"])["outputs"]
                a, b = make_format(question, inputs, outputs)
                prompts.extend(a)
                ground_truth.extend(b)
            except:
                continue
        print("done")
        assert len(prompts) == len(ground_truth)
        data_source = ["BAAI/TACO"] * len(prompts)

        def format_prompt(prompt: str) -> list[dict]:
            return [
                {
                    "role": "system",
                    "content": """
    /think
    You will be given a competitive programming problem with small input test cases. 
    You need to find the correct output for the given input by reasoning step by step through the test cases without writing any code.
    """,
                },
                {"role": "user", "content": prompt},
            ]

        print("formatting...")
        prompt = list(map(format_prompt, tqdm(prompts)))
        print("done")
        ability = ["code"] * len(prompts)

        def format_reward_model(input_output_pair: tuple[str, str]) -> dict:
            assert isinstance(input_output_pair[0], str)
            assert isinstance(input_output_pair[1], str)

            return {
                "style": "rule",
                "ground_truth": {
                    "inputs": input_output_pair[0],
                    "outputs": input_output_pair[1],
                },
            }

        reward_model = list(map(format_reward_model, zip(prompts, ground_truth)))
        df = pd.DataFrame(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": ability,
                "reward_model": reward_model,
            }
        )
        print("Writing to parquet...")
        df.to_parquet(f"TACO_{split}_processed.parquet")
        print("done")
        print(df.head())

    make_parquet("train")
    make_parquet("test")
