import json
import random

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def prepare_finetuning_data(
    input_file: str, output_file: str, model_name: str = "bert-base-uncased", max_length: int = 512
) -> None:
    """
    Prepare and tokenize data for fine-tuning a BERT model.

    Args:
        input_file (str): Path to the input JSONL file containing hero data.
        output_file (str): Path to save the processed dataset as a JSON file.
        model_name (str): Name of the pre-trained model to use for tokenization.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        None
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(input_file, "r", encoding="utf-8") as f:
        heroes = [json.loads(line) for line in f]

    print("Preparing data for fine-tuning...")
    dataset = []
    for hero in tqdm(heroes):
        if len(hero["text_content"]) == 0:
            continue
        text = f"Title: {hero['title']} Type: {hero['type']} Description: {hero['text_content']}"

        encoded = tokenizer.encode_plus(
            text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        dataset.append(
            {"input_ids": encoded["input_ids"].tolist()[0], "attention_mask": encoded["attention_mask"].tolist()[0]}
        )

    random.shuffle(dataset)

    split = int(0.9 * len(dataset))
    train_dataset = dataset[:split]
    val_dataset = dataset[split:]

    with open(output_file, "w") as f:
        json.dump({"train": train_dataset, "validation": val_dataset}, f)

    print(f"Prepared {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")


def finetune_bert(input_file: str, output_dir: str, model_name: str = "bert-base-uncased") -> None:
    """
    Fine-tune a BERT model on the prepared dataset.

    Args:
        input_file (str): Path to the processed dataset JSON file.
        output_dir (str): Directory to save the fine-tuned model and tokenizer.
        model_name (str): Name of the pre-trained model to fine-tune.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(input_file, "r") as f:
        dataset = json.load(f)

    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    class HeroDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: list) -> None:
            self.encodings = encodings

        def __getitem__(self, idx: int) -> dict:
            return {key: torch.tensor(self.encodings[idx][key]) for key in self.encodings[idx]}

        def __len__(self) -> int:
            return len(self.encodings)

    train_dataset = HeroDataset(train_dataset)
    val_dataset = HeroDataset(val_dataset)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=100,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # Prepare the finetuning data
    prepare_finetuning_data("data/characters.jsonl", "data/finetuning_data.json")

    # Print GPU memory usage if available
    if torch.cuda.is_available():
        print("GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Finetune the BERT model
    finetune_bert("data/finetuning_data.json", "hero_bert_model")
