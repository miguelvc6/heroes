import json

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def generate_embeddings(input_file, model_path, output_file):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the fine-tuned model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the hero data
    with open(input_file, "r", encoding="utf-8") as f:
        heroes = [json.loads(line) for line in f]

    # Generate embeddings
    embeddings = []
    for hero in tqdm(heroes, desc="Generating embeddings", total=len([h for h in heroes if h["text_content"] != ""])):
        if len(hero["text_content"]) == 0:
            continue
        text = f"Title: {hero['title']} Type: {hero['type']} Description: {hero['text_content']}"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Use the last hidden state of the [CLS] token as the document embedding
        embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy().tolist()[0]

        embeddings.append({"title": hero["title"], "type": hero["type"], "embedding": embedding})

    # Save the embeddings
    with open(output_file, "w") as f:
        json.dump(embeddings, f)

    print(f"Generated embeddings for {len(embeddings)} heroes.")

    return embeddings


# Usage
embeddings = generate_embeddings("data/characters.jsonl", "hero_bert_model", "data/hero_embeddings.json")
