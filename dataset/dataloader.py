import pandas as pd

pokemon_df = pd.read_csv("data/metadata_pokemon.csv")
moves_df = pd.read_csv("data/metadata_pokemon_moves.csv")

merged_df = moves_df.merge(pokemon_df, on="id")