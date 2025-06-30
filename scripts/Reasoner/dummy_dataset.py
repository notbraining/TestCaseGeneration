import pandas as pd

LENGTH = 4096 * 2


def main():
    train_data = {
        "data_source": "dummy",
        "prompt": [
            {
                "role": "user",
                "content": "What is 1 + 1?",
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "2"},
        "extra_info": {
            "split": "train",
        },
    }
    train_dataset = [train_data] * LENGTH
    train_df = pd.DataFrame(train_dataset)

    train_df.to_parquet("dummy_train.parquet", index=False)

    test_data = {
        "data_source": "dummy",
        "prompt": [
            {
                "role": "user",
                "content": "What is 1 + 1?",
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "2"},
        "extra_info": {
            "split": "test",
        },
    }
    test_dataset = [test_data] * LENGTH
    test_df = pd.DataFrame(test_dataset)

    test_df.to_parquet("dummy_test.parquet", index=False)
    print(len(train_df))


if __name__ == "__main__":
    main()
