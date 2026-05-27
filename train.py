import torch
from dataset.dataloader import get_dataloaders
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.to(device)

train_loader, val_loader, test_loader = get_dataloaders(tokenizer)

# Hyperparameters
epochs = 500
lr = 3e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
total_steps = len(train_loader) * epochs

# Scheduler for learning rate decay
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# Early stopping parameters
best_val_loss = float('inf')
patience, patience_counter = 5, 0


# Training
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_model")
        print(f"New best model saved (val loss: {avg_val_loss:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

