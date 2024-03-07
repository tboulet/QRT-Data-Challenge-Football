import pandas as pd

from src.data_loading import load_dataframe_labels, load_teamfeatures

# Read team mapping CSV
team_mapping_df = pd.read_csv('data/team_mapping.csv')

df_teamfeatures_train = load_teamfeatures({}, "data_train")
df_labels = load_dataframe_labels("data_train")

# Only use last 90% of the data. This means the first fold (10% val and 90% val) will be perfectly evaluated.
n_data_matches = df_teamfeatures_train.shape[0]
df_teamfeatures_train = df_teamfeatures_train.iloc[int(n_data_matches * 0.1) :]
df_labels = df_labels.iloc[int(n_data_matches * 0.1) :]


# Merge with team mapping dataframe to get identifiers
df_teamfeatures_train = df_teamfeatures_train.merge(team_mapping_df, how='left', left_on='HOME_TEAM_NAME', right_on='Team_Name').rename(columns={'Identifier': 'HOME_ID'})
df_teamfeatures_train = df_teamfeatures_train.merge(team_mapping_df, how='left', left_on='AWAY_TEAM_NAME', right_on='Team_Name').rename(columns={'Identifier': 'AWAY_ID'})

# Merge with labels dataframe
df_teamfeatures_train = df_teamfeatures_train.merge(df_labels, on='ID')

# Dictionary to store win rates and match counts
win_rates = {}

# Iterate over matches
for index, row in df_teamfeatures_train.iterrows():
    home_team = row['HOME_ID']
    away_team = row['AWAY_ID']
    result = 'HOME_WINS' if row['HOME_WINS'] == 1 else ('AWAY_WINS' if row['AWAY_WINS'] == 1 else 'DRAW')
    
    # Update win rates and match counts
    if (home_team, away_team) in win_rates:
        win_rates[(home_team, away_team)][result] += 1
    else:
        win_rates[(home_team, away_team)] = {'HOME_WINS': 0, 'DRAW': 0, 'AWAY_WINS': 0}
        win_rates[(home_team, away_team)][result] = 1
        
    # Do the same for the reverse match
    if result == 'HOME_WINS':
        result_reverse_match = 'AWAY_WINS'
    elif result == 'AWAY_WINS':
        result_reverse_match = 'HOME_WINS'
    elif result == 'DRAW':
        result_reverse_match = 'DRAW'
    else:
        raise ValueError('Invalid result')
        
    if (away_team, home_team) in win_rates:
        win_rates[(away_team, home_team)][result_reverse_match] += 1
    else:
        win_rates[(away_team, home_team)] = {'HOME_WINS': 0, 'DRAW': 0, 'AWAY_WINS': 0}
        win_rates[(away_team, home_team)][result_reverse_match] = 1
        
# Calculate win rates and total matches
for key, value in win_rates.items():
    total_matches = sum(value.values())
    win_rates[key]['TOTAL_MATCHES'] = total_matches
    if total_matches == 0:
        raise ValueError('Total matches is 0')
    for result in ['HOME_WINS', 'DRAW', 'AWAY_WINS']:
        win_rates[key][result + '_RATE'] = value[result] / total_matches if total_matches != 0 else 0

# Convert dictionary to dataframe
win_rates_df = pd.DataFrame(win_rates).T.reset_index().rename(columns={'level_0': 'HOME_TEAM_ID', 'level_1': 'AWAY_TEAM_ID'}).astype({'HOME_TEAM_ID': int, 'AWAY_TEAM_ID': int})

# Save to CSV
win_rates_df.to_csv('data/win_rates.csv', index=False)

print("Win rates saved to 'data/win_rates.csv'.")
