import pandas as pd
import roman
from sklearn.model_selection import train_test_split

moves_df = pd.read_csv("data/metadata_pokemon_moves.csv", encoding="utf-8")

# Data cleaning

# shorten generation column to just number and convert from roman numeral
moves_df['generation'] = moves_df['generation'].apply(lambda x: x.split(" ")[-1])
moves_df['generation'] = moves_df['generation'].apply(lambda x: roman.fromRoman(str(x)))

# fix column name 
moves_df = moves_df.rename(columns={'short_descripton': 'short_description'})

# clean short description column to remove $effect_chance%
moves_df['short_description'] = moves_df['short_description'].str.replace('$effect_chance% ', '', regex=False)

# fix empty values in power and accuracy columns
moves_df["power"] = moves_df["power"].fillna("none")
moves_df["accuracy"] = moves_df["accuracy"].fillna("none")

# print(moves_df.head(20))

# Format input strings for T5
moves_df["input_text"] = (
    "move: " + moves_df["name"].str.lower() +
    " type: " + moves_df["type"].str.lower() +
    " power: " + moves_df["power"].astype(str) +
    " accuracy: " + moves_df["accuracy"].astype(str) +
    " pp: " + moves_df["pp"].astype(str) +
    " priority: " + moves_df["priority"].astype(str) +
    " class: " + moves_df["damage_class"].str.lower() +
    " generation: " + moves_df["generation"].astype(str)
)
moves_df["target_text"] = moves_df["short_description"]

# Split into train/val/test
train_df, temp_df = train_test_split(moves_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

