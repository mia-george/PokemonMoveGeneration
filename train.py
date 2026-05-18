from dataset.dataloader import get_dataloaders
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

train_loader, val_loader, test_loader = get_dataloaders(tokenizer)

