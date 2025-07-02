import datasets
from trl import SFTConfig, SFTTrainer


def main():
    train_dataset = datasets.load_from_disk("train.hf")
    # test_dataset = datasets.load_from_disk("test.hf")

    training_args = SFTConfig(
        max_length=2048,
        output_dir="/tmp",
    )
    trainer = SFTTrainer(
        "facebook/opt-350m",
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer.train()
    return


if __name__ == "__main__":
    main()
