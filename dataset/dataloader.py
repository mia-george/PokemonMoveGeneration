import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

moves_df = pd.read_csv("data/metadata_pokemon_moves.csv", encoding="utf-8")
augmented_df = pd.read_csv("data/augmented_moves.csv", encoding="utf-8")

# Data cleaning

# fix column name 
moves_df = moves_df.rename(columns={'short_descripton': 'short_description'})

# only need these columns
moves_df = moves_df[["name", "short_description", "type", "power", "accuracy"]]

# Then combine
combined_df = pd.concat([moves_df, augmented_df], ignore_index=True)

# fix empty values in power and accuracy columns
combined_df["power"] = combined_df["power"].fillna("none")
combined_df["accuracy"] = combined_df["accuracy"].fillna("none")

# print(moves_df.head(20))
# print(f"Original unique rows: {len(moves_df)}")
# print(f"Augmented rows: {len(augmented_df)}")
# print(f"Combined: {len(combined_df)}")

# Format input strings for T5
combined_df["input_text"] = "predict move: " + combined_df["short_description"]

combined_df["target_text"] = (
    "move: " + combined_df["name"].str.lower() +
    " | type: " + combined_df["type"].str.lower() +
    " | power: " + combined_df["power"].astype(str) +
    " | accuracy: " + combined_df["accuracy"].astype(str)
)

# Split into train/val/test
train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

class MovesDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length=128, max_target_length=32):
        self.inputs = df["input_text"].tolist()
        self.targets = df["target_text"].tolist()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            self.targets[idx],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100 
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }

def get_dataloaders(tokenizer, batch_size=128):
    train_dataset = MovesDataset(train_df, tokenizer)
    val_dataset = MovesDataset(val_df, tokenizer)
    test_dataset = MovesDataset(test_df, tokenizer)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )

