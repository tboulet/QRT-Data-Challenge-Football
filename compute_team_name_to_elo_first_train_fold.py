import pandas as pd

from src.data_loading import load_dataframe_labels, load_dataframe_teamfeatures
df_teamfeatures_train = load_dataframe_teamfeatures("train")
df_labels = load_dataframe_labels("data_train")

# Only use last 1-remove_proportion% of the data.
remove_proportion = 0.5
n_data_matches = df_teamfeatures_train.shape[0]
df_teamfeatures_train = df_teamfeatures_train.iloc[int(n_data_matches * remove_proportion) :]
df_labels = df_labels.iloc[int(n_data_matches * remove_proportion) :]

# Initialize Elo ratings for each team
initial_elo = 1500  # Initial Elo rating

# Merge the two dataframes on the match identifier
merged_df = pd.merge(df_teamfeatures_train, df_labels, left_index=True, right_index=True)

# Calculate total matches played by each team
total_matches = merged_df.groupby('HOME_TEAM_NAME').size() + merged_df.groupby('AWAY_TEAM_NAME').size()

# Calculate total wins for each team
home_wins = merged_df.groupby('HOME_TEAM_NAME')['HOME_WINS'].sum()
away_wins = merged_df.groupby('AWAY_TEAM_NAME')['AWAY_WINS'].sum()
draws = merged_df.groupby('HOME_TEAM_NAME')['DRAW'].sum() + merged_df.groupby('AWAY_TEAM_NAME')['DRAW'].sum()

# Calculate total wins (including draws as 0.5 wins) for each team
total_wins = home_wins + away_wins + 0.5 * draws

# Calculate win rate for each team
win_rate = total_wins / total_matches

# Initialize Elo ratings dictionary
elo_ratings = {team: initial_elo for team in total_matches.index}

# Define K-factor for Elo updates
K_factor = 32  # You can adjust this value based on your preference

# Update Elo ratings after each match
for index, row in merged_df.iterrows():
    home_team = row['HOME_TEAM_NAME']
    away_team = row['AWAY_TEAM_NAME']
    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]
    expected_home_score = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away_score = 1 - expected_home_score
    actual_home_score = row['HOME_WINS'] + 0.5 * row['DRAW']
    actual_away_score = 1 - actual_home_score
    elo_ratings[home_team] += K_factor * (actual_home_score - expected_home_score)
    elo_ratings[away_team] += K_factor * (actual_away_score - expected_away_score)

# Create a dataframe with team names, win rates, and Elo ratings
elo_df = pd.DataFrame({'Team_Name': win_rate.index, 'global_winrate': win_rate})
elo_df['elo'] = [elo_ratings[team] for team in elo_df['Team_Name']]
elo_df['total_matches'] = total_matches

# Save the dataframe to a CSV file
elo_df.to_csv('data/elo.csv', index=False)


print("Team name to global winrate mapping saved to data/elo.csv")