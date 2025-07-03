import datasets
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def main():
    train_dataset = datasets.load_from_disk("train.hf")
    # test_dataset = datasets.load_from_disk("test.hf")

    training_args = SFTConfig(
        max_length=2048,
        output_dir="/tmp",
    )
    
    peft_config = LoraConfig(
   	r=16,
   	lora_alpha=32,
    	lora_dropout=0.05,
    	task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        "facebook/opt-350m",
        train_dataset=train_dataset,
        args=training_args,
	peft_config=pef_config
    )

    trainer.train()
    return


if __name__ == "__main__":
    main()
