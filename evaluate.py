import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from dataloader import get_dataloaders

# Load saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("saved_model")
model = T5ForConditionalGeneration.from_pretrained("saved_model")
model = model.to(device)
model.eval()

_, _, test_loader = get_dataloaders(tokenizer)

# Generate predictions
all_preds = []
all_inputs = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        all_inputs.extend(inputs)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=4, # beam search for better output quality
            early_stopping=True
        )

        # Decode generated tokens back to text
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Decode labels back to text
        labels = batch["labels"]
        labels[labels == -100] = tokenizer.pad_token_id
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_preds.extend(preds)
        all_targets.extend(targets)

# BLEU score - measures n-gram overlap between generated and reference texts
smoothing = SmoothingFunction().method1
references = [[t.split()] for t in all_targets]
hypotheses = [p.split() for p in all_preds]
bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing)

print(f"BLEU Score: {bleu_score:.4f}")

# Print some example predictions
print("\n--- Sample Predictions ---")
for i in range(5):
    print(f"\nInput:     {all_inputs[i]}")
    print(f"Generated: {all_preds[i]}")
    
    parsed = {}
    parts = all_preds[i].split(" | ")
    for part in parts:
        if ": " in part:
            key, value = part.split(": ", 1)
            parsed[key.strip()] = value.strip()
    
    print(f"Move:      {parsed.get('move', 'N/A')}")
    print(f"Type:      {parsed.get('type', 'N/A')}")
    print(f"Power:     {parsed.get('power', 'N/A')}")
    print(f"Accuracy:  {parsed.get('accuracy', 'N/A')}")
    print(f"Generation:{parsed.get('generation', 'N/A')}")
    print("-" * 40)