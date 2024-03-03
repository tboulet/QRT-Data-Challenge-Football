import pandas as pd

from src.data_loading import load_dataframe_labels, load_dataframe_teamfeatures
df_teamfeatures_train = load_dataframe_teamfeatures("train")

# Create a set of unique team names by combining home and away team names
team_names = set(df_teamfeatures_train["HOME_TEAM_NAME"]).union(
    set(df_teamfeatures_train["AWAY_TEAM_NAME"])
)

# Generate identifiers for each team name
team_mapping = {team: idx for idx, team in enumerate(sorted(team_names), start=1)}

# Create a new DataFrame to store the mapping
mapping_df = pd.DataFrame(
    list(team_mapping.items()), columns=["Team_Name", "Identifier"]
)

# Save the mapping to a CSV file
mapping_df.to_csv("data/team_mapping.csv", index=False)

print("Team name to identifier mapping saved to data/team_mapping.csv")