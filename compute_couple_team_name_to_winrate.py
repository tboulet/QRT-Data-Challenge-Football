import pandas as pd

from src.data_loading import load_teamfeatures, load_dataframe_labels

df_teamfeatures_train = load_teamfeatures({}, "data_train")
df_labels = load_dataframe_labels("data_train")


# Read team mapping CSV
team_mapping_df = pd.read_csv("data/team_mapping.csv")

# Merge with team mapping dataframe to get identifiers
df_teamfeatures_train = df_teamfeatures_train.merge(
    team_mapping_df, how="left", left_on="HOME_TEAM_NAME", right_on="Team_Name"
).rename(columns={"Identifier": "HOME_ID"})
df_teamfeatures_train = df_teamfeatures_train.merge(
    team_mapping_df, how="left", left_on="AWAY_TEAM_NAME", right_on="Team_Name"
).rename(columns={"Identifier": "AWAY_ID"})

# Merge with labels dataframe
df_teamfeatures_train = df_teamfeatures_train.merge(df_labels, on="ID")

# Dictionary to store win rates and match counts
win_rates = {}

# Iterate over matches
for index, row in df_teamfeatures_train.iterrows():
    home_team = row["HOME_ID"]
    away_team = row["AWAY_ID"]
    result = (
        "HOME_WINS"
        if row["HOME_WINS"] == 1
        else ("AWAY_WINS" if row["AWAY_WINS"] == 1 else "DRAW")
    )

    # Ensure consistent order of teams for win_rates dictionary
    if home_team > away_team:
        home_team, away_team = away_team, home_team

    # Update win rates and match counts
    if (home_team, away_team) in win_rates:
        win_rates[(home_team, away_team)][result] += 1
    else:
        win_rates[(home_team, away_team)] = {"HOME_WINS": 0, "DRAW": 0, "AWAY_WINS": 0}
        win_rates[(home_team, away_team)][result] = 1

# Calculate win rates and total matches
for key, value in win_rates.items():
    total_matches = sum(value.values())
    win_rates[key]["TOTAL_MATCHES"] = total_matches
    for result in ["HOME_WINS", "DRAW", "AWAY_WINS"]:
        win_rates[key][result + "_RATE"] = (
            value[result] / total_matches if total_matches != 0 else 0
        )

# Convert dictionary to dataframe
win_rates_df = (
    pd.DataFrame(win_rates)
    .T.reset_index()
    .rename(columns={"level_0": "HOME_TEAM_ID", "level_1": "AWAY_TEAM_ID"})
    .astype({"HOME_TEAM_ID": int, "AWAY_TEAM_ID": int})
)

# Save to CSV
win_rates_df.to_csv("data/win_rates.csv", index=False)

print("Win rates saved to 'data/win_rates.csv'.")
