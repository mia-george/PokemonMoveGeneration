import pandas as pd
import roman

moves_df = pd.read_csv("data/metadata_pokemon_moves.csv", encoding="utf-8")

# Data cleaning

# shorten generation column to just number and convert from roman numeral
moves_df['generation'] = moves_df['generation'].apply(lambda x: x.split(" ")[-1])
moves_df['generation'] = moves_df['generation'].apply(lambda x: roman.fromRoman(str(x)))

# fix column name 
moves_df = moves_df.rename(columns={'short_descripton': 'short_description'})

# clean short description column to remove $effect_chance%
moves_df['short_description'] = moves_df['short_description'].str.replace('$effect_chance% ', '', regex=False)

print(moves_df.head(20))

