import pandas as pd

from src.data_loading import load_dataframe_labels, load_dataframe_playersfeatures
df_playerfeatures_home_train, df_playerfeatures_away_train = load_dataframe_playersfeatures("train")

# Create a set of unique player names by concatenating the home and away player names
player_names = set(df_playerfeatures_home_train["PLAYER_NAME"]).union(
    set(df_playerfeatures_away_train["PLAYER_NAME"])
)
# player_names = set(df_playerfeatures_train["HOME_TEAM_NAME"]).union(
#     set(df_playerfeatures_train["AWAY_TEAM_NAME"])
# )

# Generate identifiers for each player name
player_mapping = {player: idx for idx, player in enumerate(sorted(player_names), start=1)}

# Create a new DataFrame to store the mapping
mapping_df = pd.DataFrame(
    list(player_mapping.items()), columns=["Player_Name", "Identifier"]
)

# Save the mapping to a CSV file
mapping_df.to_csv("data/player_mapping.csv", index=False)

print("Player name to identifier mapping saved to data/player_mapping.csv")