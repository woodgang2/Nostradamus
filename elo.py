import math
import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime, timedelta
# https://github.com/deepy/glicko2.git
import glicko2
from glicko2 import Player
import matplotlib.pyplot as plt
from tqdm import tqdm

home_advantage = 70
#games_df = []
#teams_df = []
#ratings_df = []
#glicko_prog_df = []

class Team:
    def __init__(self, name=None, rating=1300):
        #self.name = name
        self.rating = rating
        self.games = 1

    def update_rating(self, k_factor, score, opposing_elo):
        expected_score = self.calculate_expected_score(opposing_elo)
        self.rating += k_factor * (score - expected_score) * (1+10*math.exp (-0.35 * self.games))
        self.games += 1

    def reset_games (self):
        self.games = 1

    def calculate_expected_score(self, opposing_elo):
        return 1 / (1 + math.pow(10, (opposing_elo - self.rating) / 400))
def retrieveGames (database):
    conn = sqlite3.connect(database)
    # Write a SQL query to select all rows from the games table
    query = "SELECT * FROM games;"
    # Use read_sql_query to execute the query and store the result in a DataFrame
    games_df = pd.read_sql_query(query, conn)
    #games_df = games_df.sort_values(by='Date', ascending=True)
    # print (games_df.to_string())
    # exit(0)
    # Close the connection
    conn.close()


    games_df['Year'] = pd.to_datetime(games_df['Date']).dt.year

    # Count games for each team per year
    games_count = games_df.groupby(['Year']).apply(
        lambda x: x['Home'].value_counts() + x['Away'].value_counts()
    ).unstack(fill_value=0)

    # Filter out teams that played less than 10 games in each year
    eligible_teams_by_year = games_count.apply(lambda x: x[x >= 10].index, axis=1)

    # Function to check if both teams are eligible for a given game
    def are_teams_eligible(row):
        year = row['Year']
        home_team = row['Home']
        away_team = row['Away']
        eligible_teams = eligible_teams_by_year.get(year, [])
        return home_team in eligible_teams and away_team in eligible_teams

    # Filter games_df based on the eligibility of both teams
    filtered_games_df = games_df[games_df.apply(are_teams_eligible, axis=1)]

    # Drop the 'Year' column if it's no longer needed
    filtered_games_df = filtered_games_df.drop(columns=['Year'])

    #print (filtered_games_df.to_string())
    filtered_games_df = filtered_games_df.reset_index ()
    # print (filtered_games_df.isna().sum())
    return filtered_games_df

def pullTeams (games_df):
    unique_teams = pd.concat([games_df['Away'], games_df['Home']]).unique()
    teams_df = pd.DataFrame(unique_teams, columns=['TeamName'])
    teams_df.reset_index(inplace=True)
    teams_df.rename(columns={'index': 'TeamID'}, inplace=True)
    #print (teams_df.to_string())
    #exit (0)
    return teams_df

def average_mov (games_df):
    games_df['HomeScore'] = pd.to_numeric(games_df['HomeScore'], errors='coerce')
    games_df['AwayScore'] = pd.to_numeric(games_df['AwayScore'], errors='coerce')

    # Drop any rows that have NaN after coercion in case there were non-numeric values
    games_df = games_df.dropna(subset=['HomeScore', 'AwayScore'])

    # Calculate the margin of victory for each game
    games_df['MarginOfVictory'] = (games_df['HomeScore'] - games_df['AwayScore']).abs()

    # Calculate the average margin of victory
    average_margin = games_df['MarginOfVictory'].mean()
    print (average_margin)

def remove_rows_with_empty_scores(db_path):
    """
    Remove rows from the games table where either HomeScore or AwayScore is empty.

    Args:
    db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    delete_statement = """
    DELETE FROM games
    WHERE HomeScore IS NULL OR HomeScore = '' OR AwayScore IS NULL OR AwayScore = ''
    """
    cursor.execute(delete_statement)
    conn.commit()
    print(f"Rows with empty scores have been removed from the database.")
    conn.close()

def calculate_k_factor(margin_of_victory, elo1, elo2):
    elo_difference = math.fabs(elo1 - elo2)
    return 20 * (math.pow((2*margin_of_victory + 6), 0.8) / (7.5 + 0.024 * elo_difference))

def calculate_win_probability(rating_a, rating_b):
    exponent = (rating_b - rating_a) / 400
    expected_score_a = 1 / (1 + 10 ** exponent)
    expected_score_b = 1 - expected_score_a  # Or: 1 / (1 + 10 ** (-exponent))

    return expected_score_a, expected_score_b
def update_player (winner_elo, loser_elo, margin_of_victory):
    expected_winner = calculate_win_probability(winner_elo, loser_elo)
    expected_loser = calculate_win_probability(loser_elo, winner_elo)
    print (expected_loser, expected_winner)
    exit (0)

    k_winner = calculate_k_factor(margin_of_victory, winner_elo, loser_elo)
    k_loser = calculate_k_factor(-margin_of_victory, loser_elo, winner_elo)  # Negative MoV since it's a loss
    k_loser = k_winner

    new_winner_elo = winner_elo + k_winner * (1 - expected_winner)
    new_loser_elo = loser_elo + k_loser * (0 - expected_loser)  # S = 0 for the loser

    return new_winner_elo, new_loser_elo
def process_game(team1_name, team2_name, team1_score, team2_score, players, teams, neutral):
    """
    Process a game between two teams and update their glickos.

    Args:
    team1_name (str): The name of team 1.
    team2_name (str): The name of team 2.
    team1_score (int): The score of team 1.
    team2_score (int): The score of team 2.
    """
    # Get the Player objects for each team
    player1 = players[team1_name]
    player2 = players[team2_name]

    team1 = teams [team1_name]
    team2 = teams [team2_name]

    #print (f"{team1_name} : {team2_name}")
    #print ("1s: " , team1_score)
    #print (team2_score)
    # Convert scores to float
    team1_score = float(team1_score)
    team2_score = float(team2_score)

    # outcome1 = 0.5
    # outcome2 = 0.5
    # if (team1_score + team2_score > 0):
    #     outcome1 = team1_score / (team1_score + team2_score)
    #     outcome2 = team2_score / (team1_score + team2_score)
    margin_of_victory = math.fabs(team1_score - team2_score)
    if team1_score > team2_score:
        outcome1 = 1.0  # Team 1 won
        outcome2 = 0.0  # Team 2 lost
    elif team1_score < team2_score:
        outcome1 = 0.0  # Team 1 lost
        outcome2 = 1.0  # Team 2 won
    else:
        outcome1 = 0.5  # Draw
        outcome2 = 0.5  # Draw

    # Determine the outcome
    # if team1_score > team2_score:
    #     outcome1 = 1.0  # Team 1 won
    #     outcome2 = 0.0  # Team 2 lost
    # elif team1_score < team2_score:
    #     outcome1 = 0.0  # Team 1 lost
    #     outcome2 = 1.0  # Team 2 won
    # else:
    #     outcome1 = 0.5  # Draw
    #     outcome2 = 0.5  # Draw
    #     #are there rainout called draws in college baseball? Just in case

    # Update player ratings
    #print (player1.rating)
    # awayRating = player1.rating - 72
    # homeRating = player2.rating + 72

    original_elo1 = team1.rating - home_advantage + neutral
    original_elo2 = team2.rating + home_advantage - neutral
    k=calculate_k_factor (margin_of_victory, original_elo1, original_elo2+home_advantage+neutral)
    # original_elo1 -= home_advantage
    original_rating1 = player1.rating - home_advantage - neutral
    original_rd1 = player1.rd
    player2.rating += home_advantage
    team1.update_rating (k, outcome1, original_elo2)
    player1.update_player([player2.rating], [player2.rd], [outcome1])
    player2.rating -= home_advantage
    team2.update_rating (k, outcome2, original_elo1)
    # original_elo1 += home_advantage
    player2.update_player([original_rating1], [original_rd1], [outcome2])
    #print (player2.rating)
    # Example usage:
    # process_game('Team A', 'Team B', 3, 2)

    updated_elo = [(team_name, player.rating) for team_name, player in teams.items()]
    # After processing all games, you can create a DataFrame to display updated ratings
    updated_ratings = [(team, player.rating, player.rd) for team, player in players.items()]
    elo_df = pd.DataFrame(updated_elo, columns=['Team', 'Rating'])
    ratings_df = pd.DataFrame(updated_ratings, columns=['Team', 'Rating', 'Rating Deviation'])
    # elo_df = elo_df.merge(ratings_df[['Team', 'Rating Deviation']], on='Team', how='left')
    # elo_df ['Rating Deviation'] = elo_df ['Rating Deviation'].apply(lambda x: '{:.2f}'.format(x))
    return ratings_df, elo_df
    #print(ratings_df)

vCount = 0
def createRatingTable (games_df, players, teams):
    ratings_df = []
    glicko_changes = []
    last_date = None  # Keep track of the last date processed

    for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Computing Glicko..."):
        current_date = pd.to_datetime(row['Date'])  # Assuming 'Date' is in a format pd.to_datetime can parse
        if last_date is not None and current_date.year != last_date.year:
            # We have progressed to a new year, regress each team's rating
            for player in players.values():
                player.rating = 0.75 * player.rating + 0.25 * 1505
        last_date = current_date
        #ratings_df = process_game(row['Away'], row['Home'], row['AwayScore'], row['HomeScore'])
        pregame_glicko_away = players[row['Away']].rating
        pregame_glicko_home = players[row['Home']].rating

        global vCount

        # Process the game to update glicko ratings
        if (row ['Home'] in row ['Site']):
            ratings_df,_ = process_game(row['Away'], row['Home'], row['AwayScore'], row['HomeScore'],players , teams, 0)
        elif (row ['Away'] in row ['Site']):
            ratings_df,_ = process_game(row['Home'], row['Away'], row['HomeScore'], row['AwayScore'],players , teams, 0)
        else:
            ratings_df,_ = process_game(row['Away'], row['Home'], row['AwayScore'], row['HomeScore'],players , teams, -home_advantage)
        if (row ['Home'] == "Virginia" or row ['Away'] == "Virginia"):
            vCount += 1
        # Retrieve postgame glicko ratings
        postgame_glicko_away = players[row['Away']].rating
        postgame_glicko_home = players[row['Home']].rating

        # if (row ['Home'] == "Virginia"):
        #     print (row)
        #     print (row ['Date'], pregame_glicko_home)
        #     print (row ['Date'], postgame_glicko_home)
        #     print (vCount)
        # if (row ['Away'] == "Virginia"):
        #     print (row)
        #     print (row ['Date'], pregame_glicko_away)
        #     print (row ['Date'], postgame_glicko_away)
        #     print (vCount)


        # Append the pregame and postgame glicko ratings to the list
        glicko_changes.append({
            'pregame_glicko_away': pregame_glicko_away,
            'pregame_glicko_home': pregame_glicko_home,
            'postgame_glicko_away': postgame_glicko_away,
            'postgame_glicko_home': postgame_glicko_home

        })
    glicko_changes_df = pd.DataFrame(glicko_changes)
    games_df = pd.concat([games_df, glicko_changes_df], axis=1)
    ratings_df['SortKey'] = ratings_df['Rating Deviation'].apply(lambda x: 0 if x <= 110 else 1)
    ratings_df = ratings_df.sort_values(by=['SortKey', 'Rating'], ascending=[True, False])
    ratings_df.drop('SortKey', axis=1, inplace=True)
    ratings_df['Rank'] = ratings_df.reset_index().index + 1

    win_loss_records = {team: {'wins': 0, 'losses': 0} for team in ratings_df['Team']}

    # Iterate through games_df to update win-loss records
    #print (games_df.to_string ())
    for index, row in games_df.iterrows():
        #print (row)
        away_team, home_team = row['Away'], row['Home']
        away_score, home_score = int(row['AwayScore']), int(row['HomeScore'])

        # Determine the winner and loser
        if away_score > home_score:
            win_loss_records[away_team]['wins'] += 1
            win_loss_records[home_team]['losses'] += 1
        elif home_score > away_score:
            win_loss_records[home_team]['wins'] += 1
            win_loss_records[away_team]['losses'] += 1
        # Assuming no ties, but you can add an else case to handle ties if necessary

    # Create a new column in ratings_df for the win-loss record
    ratings_df['Record'] = ratings_df['Team'].apply(
        lambda team: f"{win_loss_records[team]['wins']}-{win_loss_records[team]['losses']}"
    )
    columns_except_rank = [col for col in ratings_df.columns if col != 'Rank']
    ratings_df = ratings_df[['Rank'] + columns_except_rank]
    return ratings_df, games_df

def createEloRatingTable (games_df, players, ignore_this):
    ratings_df = []
    elo_changes = []
    last_date = None  # Keep track of the last date processed
    games_df['New_Date'] = pd.to_datetime(games_df['Date'])
    win_loss_by_year = {}
    for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Computing Elo..."):
        current_date = pd.to_datetime(row['Date'])  # Assuming 'Date' is in a format pd.to_datetime can parse
        if last_date is not None and current_date.year != last_date.year:
        # We have progressed to a new year, regress each team's rating
            for player in players.values():
                player.rating = 0.75 * player.rating + 0.25 * 1505
                player.games = 1
        last_date = current_date
        #ratings_df = process_game(row['Away'], row['Home'], row['AwayScore'], row['HomeScore'])
        pregame_elo_away = players[row['Away']].rating
        pregame_elo_home = players[row['Home']].rating
        global vCount

        postgame_elo_home = 0
        postgame_elo_away = 0
        # Process the game to update glicko ratings
        if (row ['Home'] in row ['Site']):
            _,ratings_df = process_game(row['Away'], row['Home'], row['AwayScore'], row['HomeScore'],ignore_this , players, 0)
        elif (row ['Away'] in row ['Site']):
            _,ratings_df = process_game(row['Home'], row['Away'], row['HomeScore'], row['AwayScore'],ignore_this , players, 0)
        else:
            _,ratings_df = process_game(row['Away'], row['Home'], row['AwayScore'], row['HomeScore'],ignore_this , players, -home_advantage)
        if (row ['Home'] == "Virginia" or row ['Away'] == "Virginia"):
            vCount += 1
        # Retrieve postgame glicko ratings
        postgame_elo_away = players[row['Away']].rating
        postgame_elo_home = players[row['Home']].rating

        # if (row ['Home'] == "Virginia"):
        #     print (row)
        #     print (row ['Date'], pregame_glicko_home)
        #     print (row ['Date'], postgame_glicko_home)
        #     print (vCount)
        # if (row ['Away'] == "Virginia"):
        #     print (row)
        #     print (row ['Date'], pregame_glicko_away)
        #     print (row ['Date'], postgame_glicko_away)
        #     print (vCount)


        # Append the pregame and postgame glicko ratings to the list
        elo_changes.append({
            'pregame_elo_away': pregame_elo_away,
            'pregame_elo_home': pregame_elo_home,
            'postgame_elo_away': postgame_elo_away,
            'postgame_elo_home': postgame_elo_home

        })
        # print (elo_changes)
        # exit(0)
    elo_changes_df = pd.DataFrame(elo_changes)
    # print(elo_changes_df)
    games_df = pd.concat([games_df, elo_changes_df], axis=1)
    # print (games_df)
    ratings_df = ratings_df.sort_values(by=['Rating'], ascending=[False])
    # print (ratings_df)
    ratings_df['Rank'] = ratings_df.reset_index().index + 1

    win_loss_records = {team: {'wins': 0, 'losses': 0} for team in ratings_df['Team']}

    # Iterate through games_df to update win-loss records
    for index, row in games_df.iterrows():
        year = row['New_Date'].year
        away_team, home_team = row['Away'], row['Home']
        away_score, home_score = int(row['AwayScore']), int(row['HomeScore'])

        for team in [home_team, away_team]:
            if team not in win_loss_records:
                win_loss_records[team] = {}
            if year not in win_loss_records[team]:
                win_loss_records[team][year] = {'wins': 0, 'losses': 0}

    # Determine the winner and loser
        if home_score > away_score:
            win_loss_records[home_team][year]['wins'] += 1
            win_loss_records[away_team][year]['losses'] += 1
        elif away_score > home_score:
            win_loss_records[away_team][year]['wins'] += 1
            win_loss_records[home_team][year]['losses'] += 1
        # Assuming no ties, but you can add an else case to handle ties if necessary

    # Create a new column in ratings_df for the win-loss record
    #ratings_df['Record'] = ratings_df['Team'].apply(
    #     lambda team: f"{win_loss_records[team]['wins']}-{win_loss_records[team]['losses']}"
    # )
    for year in sorted({row['New_Date'].year for _, row in games_df.iterrows()}, reverse=True):
        win_loss_column = f'Record_{year}'
        ratings_df[win_loss_column] = [
            f"{win_loss_records[team][year]['wins']}-{win_loss_records[team][year]['losses']}"
            if year in win_loss_records[team] else "" for team in ratings_df['Team']
        ]
    columns_except_rank = [col for col in ratings_df.columns if col != 'Rank']
    ratings_df = ratings_df[['Rank'] + columns_except_rank]
    # print (ratings_df)
    # print (games_df)
    #games_df = games_df.drop ('New_Date')
    return ratings_df, games_df

def create_elo_progression_table (teams_df, games_df):
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2023, 6, 26)
    date_range = pd.date_range(start=start_date, end=end_date)

    teams_initial_elo = {team: [1300 if date == start_date else None for date in date_range] for team in teams_df['TeamName']}
    elo_prog_df = pd.DataFrame(teams_initial_elo, index=date_range)
    for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Saving Elo Progression..."):
        game_date = pd.to_datetime(row['Date'])
        if game_date in elo_prog_df.index:
            elo_prog_df.at[game_date, row['Away']] = row['postgame_elo_away']
            elo_prog_df.at[game_date, row['Home']] = row['postgame_elo_home']
    elo_prog_df.ffill(inplace=True)
    # glicko_prog_df2 = glicko_prog_df.loc[:, ['Virginia']]
    #glicko_prog_df2 = glicko_prog_df.loc[:, ['Lafayette']]
    #print (glicko_prog_df2.to_string ())
    #exit (0)
    return elo_prog_df;

def create_glicko_progression_table (teams_df, games_df):
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2023, 6, 26)
    date_range = pd.date_range(start=start_date, end=end_date)

    teams_initial_glicko = {team: [1500 if date == start_date else None for date in date_range] for team in teams_df['TeamName']}
    glicko_prog_df = pd.DataFrame(teams_initial_glicko, index=date_range)
    for index, row in tqdm(games_df.iterrows(), total=games_df.shape[0], desc="Saving Glicko Progression..."):
        game_date = pd.to_datetime(row['Date'])
        if game_date in glicko_prog_df.index:
            glicko_prog_df.at[game_date, row['Away']] = row['postgame_glicko_away']
            glicko_prog_df.at[game_date, row['Home']] = row['postgame_glicko_home']
    glicko_prog_df.ffill(inplace=True)
    # glicko_prog_df2 = glicko_prog_df.loc[:, ['Virginia']]
    #glicko_prog_df2 = glicko_prog_df.loc[:, ['Lafayette']]
    #print (glicko_prog_df2.to_string ())
    #exit (0)
    return glicko_prog_df;

#print(glicko_prog_df)
#lines = glicko_prog_df.plot.line()

def plotAll (teams_df, glicko_prog_df):
    start_date = datetime(2023, 2, 16)
    days_since_start = (glicko_prog_df.index - start_date).days


    plt.figure(figsize=(20, 20))

    for team in teams_df['TeamName']:
        plt.plot(days_since_start, glicko_prog_df[team], label=team)

    plt.title('Ratings Over Time')
    plt.xlabel('Days Since Season Start')
    plt.ylabel('Rating')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.show()

def normalize_rgb(color):
    """Normalize RGB values to the range 0 to 1."""
    return tuple(c / 255 for c in color)
def plotTeams(team_names, rating_prog_df, team_colors=None, year = None):
    """
    Generates a line graph of the specified teams' glicko ratings over the season with optional line colors specified as RGB.

    Args:
    team_names (list of str): A list of team names to plot. Accepts 1-5 teams.
    glicko_prog_df (DataFrame): DataFrame containing the glicko ratings.
    team_colors (list of tuple, optional): A list of RGB tuples for each team's line plot. Defaults to None.
    """
    # Convert the index to the number of days since the season began

    start_date = datetime(2010, 1, 1)
    rating_prog_df['Date'] = [start_date + timedelta(days=x) for x in range(rating_prog_df.shape[0])]

    # Convert 'Date' to datetime and filter out rows not in January to June
    rating_prog_df['Date'] = pd.to_datetime(rating_prog_df['Date'])
    if year is not None:
        rating_prog_df = rating_prog_df[rating_prog_df.index.year == year]
        rating_prog_df = rating_prog_df[rating_prog_df.index.month <= 6] # Reset start_date to the beginning of the specified year

    #rating_prog_df = rating_prog_df[rating_prog_df['Date'].dt.month <= 6]

    # Reset the index after filtering
    #days_since_start = (rating_prog_df['Date'] - start_date).dt.days
    #rating_prog_df.reset_index(drop=True, inplace=True)

    #start_date = datetime(2023, 2, 16)
    #days_since_start = (rating_prog_df.index)

    plt.figure(figsize=(10, 6))

    if team_colors is None:
        team_colors = [None] * len(team_names)  # Default to None if no colors are provided
    for team_name, color in zip(team_names, team_colors):
        if team_name not in rating_prog_df.columns:
            print(f"Team '{team_name}' not found.")
            continue
        color = normalize_rgb(color) if color else None
        # Plotting each specified team's glicko rating over time with specified RGB color
        plt.plot(rating_prog_df.index, rating_prog_df[team_name], label=team_name, marker='o', linestyle='-', markersize=4, color=color)

    plt.title('Ratings Over Time for Selected Teams')
    plt.xlabel('Month')
    plt.ylabel('Rating')
    # plt.ylim([1500, 2050])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_log_loss(probability, actual_result):
    # Ensuring no probability is exactly 0 or 1 for log loss calculation
    epsilon = 1e-15
    probability = np.clip(probability, epsilon, 1 - epsilon)
    if actual_result == 1:
        return -np.log(probability)
    else:
        return -np.log(1 - probability)
def checkAlgo (games_df):
    # games_df['higher_glicko_won'] = (
    #         ((games_df['pregame_glicko_home'] + home_advantage > games_df['pregame_glicko_away']) & (games_df['HomeScore'] > games_df['AwayScore'])) |
    #         ((games_df['pregame_glicko_home'] + home_advantage < games_df['pregame_glicko_away']) & (games_df['HomeScore'] < games_df['AwayScore']))
    # )
    # games_df['higher_elo_won'] = (
    #         ((games_df['pregame_elo_home'] + home_advantage > games_df['pregame_elo_away']) & (games_df['HomeScore'] > games_df['AwayScore'])) |
    #         ((games_df['pregame_elo_home'] + home_advantage < games_df['pregame_elo_away']) & (games_df['HomeScore'] < games_df['AwayScore']))
    # )
    modified_games_df = games_df
    # modified_games_df['New Date'] = pd.to_datetime(games_df['Date'])
    # modified_games_df  = modified_games_df[modified_games_df['New Date'].dt.year == 2023]
    modified_games_df ['higher_glicko_won'] = None
    modified_games_df ['higher_elo_won'] = None
    modified_games_df ['elo_log_loss'] = None
    for index, row in modified_games_df.iterrows():
        if row['Home'] in row['Site']:
            modified_games_df.loc[index, 'higher_glicko_won'] = row['pregame_glicko_home'] + home_advantage > row['pregame_glicko_away']
            modified_games_df.loc[index, 'higher_elo_won'] = row['pregame_elo_home'] + home_advantage > row['pregame_elo_away']
            elo_win_prob,_ = calculate_win_probability ( row['pregame_elo_home'] + home_advantage, row['pregame_elo_away'])
            #print (row ['pregame_elo_home'] + home_advantage, row['pregame_elo_away'], a, elo_win_prob)
            #exit (0)
            modified_games_df ['elo_log_loss'] = calculate_log_loss(elo_win_prob,1)
        elif row['Away'] in row['Site']:
            modified_games_df.loc[index, 'higher_glicko_won'] = row['pregame_glicko_home'] - home_advantage > row['pregame_glicko_away']
            modified_games_df.loc[index, 'higher_elo_won'] = row['pregame_elo_home'] - home_advantage > row['pregame_elo_away']
            elo_win_prob,_ = calculate_win_probability ( row['pregame_elo_home'] - home_advantage, row['pregame_elo_away'])
            modified_games_df ['elo_log_loss'] = calculate_log_loss(elo_win_prob,1)
        elif 'eutral' in row['Site']:
            modified_games_df.loc[index, 'higher_glicko_won'] = row['pregame_glicko_home'] > row['pregame_glicko_away']
            modified_games_df.loc[index, 'higher_elo_won'] = row['pregame_elo_home'] > row['pregame_elo_away']
            elo_win_prob,_ = calculate_win_probability ( row['pregame_elo_home'], row['pregame_elo_away'])
            modified_games_df ['elo_log_loss'] = calculate_log_loss(elo_win_prob,1)
        else:
            print ("Site?")
            print (row)
            exit (0)
    # Calculate the percentage of times the team with higher pregame glicko won
    higher_glicko_win_percentage = modified_games_df['higher_glicko_won'].mean() * 100
    higher_elo_win_percentage = modified_games_df['higher_elo_won'].mean() * 100

    print(f"The team with the higher pregame elo won {higher_elo_win_percentage:.2f}% of the time.")
    print(f"The team with the higher pregame glicko won {higher_glicko_win_percentage:.2f}% of the time.")
    print (f"Average elo log loss {modified_games_df['elo_log_loss'].mean ()}")
    # exit (0)
#plotTeams (["Virginia", "Virginia Tech", "Clemson", "Florida", "Wake Forest", "Vanderbilt", "LSU", "Arkansas", "Miami"])
#plotTeams (['Virginia', 'Stanford'])

def scrapeGames (start_year, start_month, start_day, end_year, end_month, end_day, delay):
    # Define the start and end date for the range
    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day)

    # Create a date range
    date_range = pd.date_range(start_date, end_date)

    # Create an empty list to hold the game data
    game_data_list = []

    i = 0
    # Iterate over each date in the date range
    for single_date in date_range:
        # Format the date as YYYYMMDD
        formatted_date = single_date.strftime('%Y%m%d')
        url = f'https://d1baseball.com/scores/?date={formatted_date}'

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0"
        }

        # Make the request and parse the HTML
        soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")

        # Extract game data for the current date
        for s in soup.select(".box-score"):
            #print ("hi")
            name1, name2 = [n.get_text(strip=True) for n in s.select(".team-title")]
            score1, score2 = [n.get_text(strip=True) for n in s.select(".team-score")]
            # Make sure there are enough score elements before unpacking
            #if len(scores) >= 4:
                #score1, score2 = scores[0], scores[3]  # Assuming these are the correct indices for scores

                # Use the single_date variable for the game date
            game_data = {
                'Date': single_date.date(),
                'Away': name1,
                'AwayScore': score1,
                'HomeScore': score2,
                'Home': name2
            }
            game_data_list.append(game_data)
            #print (game_data)
        print (i)
        i += 1
        time.sleep (delay)
    # Create a DataFrame from the list of game data
    games_df = pd.DataFrame(game_data_list)

    # Print the DataFrame
    print(games_df.to_string())

    # Save to SQLite database
    conn = sqlite3.connect('Data/ncaa_games_since_2000')
    games_df.to_sql('games', conn, if_exists='replace', index=False)
    conn.close()


def simMatch (month, day, away_team, home_team, glicko_prog_df, neutralStadium=None):
    try:
        date =  datetime(2023, month, day)
    except:
        print (f"2023/{month}/{day} is an invalid date. Please input a day between 2/16 and 5/28")
        exit (0)
    try:
        away_rating = glicko_prog_df.loc [f'2023/{month}/{day}', away_team]
    except:
        print (f"{away_team} not found")
        exit (0)
    try:
        home_rating = glicko_prog_df.loc [f'2023/{month}/{day}', home_team]
    except:
        print (f"{home_team} not found")
        exit (0)
    probability_a, probability_b = calculate_win_probability (away_rating, home_rating + home_advantage)
    isAway = "(away)"
    isHome = "(home)"
    if (neutralStadium == 1):
        probability_a, probability_b = calculate_win_probability (away_rating, home_rating)
        isAway = "(neutral site)"
        isHome = "(neutral site)"
    start_date = datetime(2023, 2, 16)
    days_since_start = (date - start_date).days

    away_team = ('{: <25}'.format(away_team))
    home_team = ('{: <25}'.format(home_team))

    print (f"Match 2023/{month}/{day} ({days_since_start} days since start of season)")
    print(f"    Probability {away_team} {'{0:.2f}'.format(away_rating)} {isAway} wins: {probability_a * 100:.2f}%")
    print(f"    Probability {home_team} {'{0:.2f}'.format(home_rating)} {isHome} wins: {probability_b * 100:.2f}%")
    return [probability_a, probability_b]

hasBeenSetUp = 0
def set_up ():
    column_names = ['Date', 'Home', 'HomeScore', 'Away', 'AwayScore', 'Site']
    dtype_spec = {'HomeScore': int, 'AwayScore': int}
    #names=column_names, header=0 if header_present else None, skiprows=1 if header_present else 0)
    df = pd.read_csv('scores.csv', names=column_names, dtype=dtype_spec)
    # df = pd.read_csv('scores.csv')
    #
    # # Step 2 & 3: Create a SQLite database and connect to it
    conn = sqlite3.connect('Data/ncaa_games_since_2000')
    #
    # # Step 4: Write the DataFrame to the SQLite database
    df.to_sql('games', conn, if_exists='replace', index=False)
    #
    # # Close the connection to the database
    conn.close()

    global hasBeenSetUp
    if (hasBeenSetUp == 0):
        print ("Setting up, hang tight!")
    games_df = retrieveGames('Data/ncaa_games_since_2000')
        #print ("Error setting up, scraping games...")
        #scrapeGames(2023, 2, 17, 2023, 6, 26, 0.5)
        #print ("Finished scraping")
    print ("Games loaded")
    teams_df = pullTeams(games_df)
    print ("Teams loaded")
    players = {row['TeamName']: Player() for index, row in teams_df.iterrows()}
    #print ("Computing glicko...")
    teams = {row['TeamName']: Team() for index, row in teams_df.iterrows()}
    #print ("Computing elo...")
    time.sleep (0.5)
    ratings_df, games_df = createRatingTable(games_df, players, teams)
    time.sleep (0.5)
    print ("Finished computing glicko")
    time.sleep (0.5)
    elo_df, games_df = createEloRatingTable(games_df, teams, players)
    print ("Finished computing elo")
    #print(games_df)
    #print (elo_df.to_string())
    #exit (0)
    ratings_df_display = ratings_df
    ratings_df_display ['Rating'] = ratings_df['Rating'].apply(lambda x: '{:.2f}'.format(x))
    ratings_df_display ['Rating Deviation'] = ratings_df['Rating Deviation'].apply(lambda x: '{:.2f}'.format(x))
    elo_df_display = elo_df
    elo_df_display ['Rating'] = elo_df['Rating'].apply(lambda x: '{:.2f}'.format(x))
    glicko_prog_df = create_glicko_progression_table(teams_df, games_df)
    time.sleep (0.25)
    print ("Day to day glicko values saved")
    time.sleep (0.25)
    elo_prog_df = create_elo_progression_table(teams_df, games_df)
    time.sleep (0.25)
    print ("Day to day elo values saved")
    time.sleep (0.5)
    if (hasBeenSetUp == 0):
        print (games_df)
        print (teams_df)
        print ("ELO Rankings")
        print (elo_df_display.to_string())
        #print ("Glicko Rankings")
        #print (ratings_df_display.to_string())
        #print (games_df)
        print (glicko_prog_df)
        print (elo_prog_df)
        checkAlgo(games_df)
        # exit (0)
    hasBeenSetUp = 1
    #for column in glicko_prog_df.columns:
        #print(column)
    #exit (0)
    print ("Set up finished")
    return games_df, teams_df, ratings_df, elo_df, glicko_prog_df, elo_prog_df

def autoCalc_simMatch (month, day, away_team, home_team, neutral_site=None, glicko_override = None):
    games_df, teams_df, ratings_df, elo_df, glicko_prog_df, elo_prog_df = set_up ()
    if (glicko_override == 1):
        simMatch (month, day, away_team, home_team, glicko_prog_df, neutral_site)
    else:
        simMatch (month, day, away_team, home_team, elo_prog_df, neutral_site)

def autoCalc_plotTeams (team_names, team_colors=None, glicko_override = None):
    #if team_colors is None:
        #team_colors = [''] * len(team_names)
    games_df, teams_df, ratings_df, elo_df, glicko_prog_df, elo_prog_df = set_up ()
    if (glicko_override == 1):
        plotTeams (team_names, glicko_prog_df, team_colors)
    else:
        plotTeams (team_names, elo_prog_df, team_colors)

def printTeamSchedule (team_name, games_df):
    print (f"{team_name} Season Summary")
    """
    Print the complete list of games for a specified team.

    Args:
    team_name (str): The name of the team.
    games_df (DataFrame): DataFrame containing game records.
    """
    # Filter games where the team is playing at home or away
    past_glicko = 1500
    past_elo = 1300
    home_games = games_df[games_df['Home'] == team_name]
    away_games = games_df[games_df['Away'] == team_name]
    team_games = pd.concat([home_games, away_games]).sort_index ();
    #print (team_games.sort_index ().to_string())
    #team_games = team_games.sort_values('Date')
    wins = []
    losses = []
    for index, row in team_games.iterrows():
        #print (row)
        date = row['Date']
        home_team = row['Home']
        away_team = row['Away']
        home_score = int (row['HomeScore'])
        away_score = int (row['AwayScore'])
        glicko = row ['postgame_glicko_home']
        opponent_glicko = row ['pregame_glicko_away']
        elo = row ['postgame_elo_home']
        opponent_elo = row ['postgame_elo_away']
        vs = 'vs.'
        if team_name == home_team:
            #print ("HOME___________________________________")
            opponent = away_team
            team_score, opponent_score = home_score, away_score
            _,win_probability = calculate_win_probability(opponent_elo, elo+home_advantage)
        else:
            opponent = home_team
            team_score, opponent_score = away_score, home_score
            glicko = row ['postgame_glicko_away']
            opponent_glicko = row ['pregame_glicko_home']
            elo = row ['postgame_elo_away']
            opponent_elo = row ['pregame_elo_home']
            _,win_probability = calculate_win_probability(opponent_elo+home_advantage, elo)
        if ('neutral' in row ['Site']):
            vs = 'n'
        elif (team_name in row ['Site']):
            vs = 'vs.'
        else:
            vs = '@'
        result = 'W'
        start_bold = '\033[1m'
        end_bold = '\033[0m'
        start_bold_opponent = ''
        end_bold_opponent = ''
        if (team_score < opponent_score):
            #print (row)
            start_bold = ''
            end_bold = ''
            start_bold_opponent = '\033[1m'
            end_bold_opponent = '\033[0m'
            result = 'L'
        glicko_change = "{:+.2f}".format(round(glicko - past_glicko, 2))
        elo_change = "{:+.2f}".format(round(elo - past_elo, 2))
        #glicko_change = '{0:.2f}'.format(glicko_change)
        #glicko_display = '{0:.2f}'.format(glicko)
        #glicko = '{0:.2f}'.format(glicko)
        team_score = ('{: <2}'.format(team_score))
        opponent_score = ('{: >2}'.format(opponent_score))
        vs = ('{:^3}'.format(vs))
        print(f"    {date} {result} {team_name}  {start_bold}{team_score}{end_bold}  {vs}  {start_bold_opponent}{opponent_score}{end_bold_opponent}  {opponent}, New elo: {'{0:.2f}'.format(elo)} ({elo_change}) New glicko: {'{0:.2f}'.format(glicko)} ({glicko_change})")
        # print(f"Date: {date}, {start_bold}{team_name}{end_bold} {vs} {opponent}, Score: {team_score}-{opponent_score} {result}, New glicko: {glicko} ({glicko_change})")
        past_glicko = glicko
        past_elo = elo
        if result == 'W':
            wins.append((date, vs, opponent, opponent_elo, elo, win_probability))
        else:
            losses.append((date, vs, opponent, opponent_elo, elo, win_probability))

    wins.sort(key=lambda x: x[3], reverse=True)  # Highest glicko first
    losses.sort(key=lambda x: x[3])  # Lowest glicko first
    print("\n   Highest-Rated Wins (by opponent rating):")
    printed_teams = set()
    count = 0

    for win in wins:
        if win[2] not in printed_teams and count < 3:  # win[2] should contain the opponent's team name
            # _, win_probability = calculate_win_probability(win[4], win[3])
            # win_probability = '{0:.2f}'.format(win_probability)
            print(f"        {win[0]} {win[1]} {win[2]} with Elo {'{0:.2f}'.format(win[3])} (Win Probability: {win[5] * 100:.2f}%)")
            printed_teams.add(win[2])  # Add the team to the set of printed teams
            count += 1  # Increment the count of printed wins
    # Print biggest losses
    print("\n   Lowest-Rated Losses (by opponent rating):")
    printed_teams = printed_teams.clear ()
    printed_teams = set ()
    count = 0
    # for loss in losses[:3]:
    for loss in losses:
        if loss[2] not in printed_teams and count < 3:
            # _, win_probability = calculate_win_probability(loss [4], loss [3])
            print(f"        {loss [0]} {loss [1]} {loss[2]} with Elo {'{0:.2f}'.format(loss [3])} (Win Probability: {loss [5] * 100:.2f}%)")
            printed_teams.add (loss [2])
            count += 1
        if (count > 2):
            break
    print ()

def autoCalc_printTeamSchedule (tean_name):
    games_df, teams_df, ratings_df, elo_df, glicko_prog_df, elo_prog_df = set_up ()
    # print (games_df)
    printTeamSchedule (tean_name, games_df)

def separate_rank(team):
    match = re.match(r'(\d+)\s+(.*)', team)
    if match:
        return match.group(1), match.group(2)  # rank, team name
    else:
        return '', team

def clean_data ():
    # Connect to the SQLite database
    conn = sqlite3.connect('Data/ncaa_games_since_2000')

    # Read the game records into a pandas DataFrame
    query = 'SELECT * FROM games'  # Assuming the table is named 'games'
    games_df = pd.read_sql_query(query, conn)
    for index, row in games_df.iterrows():
        away_rank, away_team = separate_rank(row['Away'])
        home_rank, home_team = separate_rank(row['Home'])

        # Update the DataFrame with the separated values
        games_df.at[index, 'Away'] = away_team
        games_df.at[index, 'Home'] = home_team
        games_df.at[index, 'away_rank'] = away_rank
        games_df.at[index, 'home_rank'] = home_rank
    games_df['away_rank'] = games_df['away_rank'].replace('', None).astype(float)
    games_df['home_rank'] = games_df['home_rank'].replace('', None).astype(float)
    #print (games_df.to_string())
    # Close the database connection
    games_df.to_sql('games', conn, if_exists='replace', index=False)
    conn.close()


# autoCalc_simMatch (4,28,"LSU", "Florida", 0)
# autoCalc_plotTeams (['LSU', 'Florida'])
#autoCalc_plotTeams (['Canisius', 'Manhattan'])
#autoCalc_printTeamSchedule ('Utah Tech')
#autoCalc_printTeamSchedule ('Virginia')
#autoCalc_plotTeams (['Georgia Tech', 'Virginia', 'Pittsburgh'])
#scrapeGames (2023, 2, 17, 2023, 6, 28, 1)
#games_df, teams_df, ratings_df, glicko_prog_df = setUp ()
#simMatch (4,28,"LSU", "Southeast Missouri State", 0, glicko_prog_df)
#simMatch (4,28,"LSU", "Virginia", 0, glicko_prog_df)
#simMatch (4,28,"LSU", "Virginia", 1, glicko_prog_df)

import sqlite3

import sqlite3

def remove_duplicate_games(db_path):
    """
    Remove duplicate rows in the games table based on Date, Away, Home, AwayScore, and HomeScore columns.

    Args:
    db_path (str): Path to the SQLite database file.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Create a new table with unique rows
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS unique_games AS
    SELECT DISTINCT Date, Away, Home, AwayScore, HomeScore
    FROM games
    """)

    # Step 2: Delete the original games table (consider backing up before doing this)
    cursor.execute("DROP TABLE games")

    # Step 3: Rename the new table to the original table's name
    cursor.execute("ALTER TABLE unique_games RENAME TO games")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Duplicate rows based on Date, Away, Home, AwayScore, and HomeScore have been removed.")

def copy_game_records(source_db, dest_db):
    """
    Copy rows from the source database to the destination database for specific columns,
    setting any unspecified columns in the destination to NULL.

    Args:
    source_db (str): Path to the source SQLite database file.
    dest_db (str): Path to the destination SQLite database file.
    """
    # Connect to the source database and read the specified columns into a DataFrame
    with sqlite3.connect(source_db) as src_conn:
        query = "SELECT Date, Away, AwayScore, HomeScore, Home FROM games"
        games_df = pd.read_sql_query(query, src_conn)

    # Connect to the destination database
    with sqlite3.connect(dest_db) as dest_conn:
        cursor = dest_conn.cursor()

        # Dynamically create the list of column names based on the DataFrame
        columns = games_df.columns.tolist()
        placeholders = ", ".join(["?"] * len(columns))  # Create placeholders for parameters

        # Assuming the destination table has the same name 'games' and potentially more columns
        insert_stmt = f"INSERT INTO games ({', '.join(columns)}) VALUES ({placeholders})"

        # For columns not present in the source, we assume they are handled as NULL by default in the database schema

        # Execute the insert statement for each row in the DataFrame
        for _, row in games_df.iterrows():
            cursor.execute(insert_stmt, tuple(row))

        dest_conn.commit()

def update_team_names(db_path, old_name, new_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Prepare the SQL UPDATE statements for the "Home" and "Away" columns
    update_home = "UPDATE games SET Home = ? WHERE Home = ?"
    update_away = "UPDATE games SET Away = ? WHERE Away = ?"

    # Execute the UPDATE statements
    cursor.execute(update_home, (new_name, old_name))
    cursor.execute(update_away, (new_name, old_name))

    # Commit the changes
    conn.commit()

    # Close the database connection
    conn.close()

    print(f"Team names updated from '{old_name}' to '{new_name}' in both 'Home' and 'Away' columns.")

def remove_matches_involving_team(db_path, team_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    delete_statement = """
    DELETE FROM games
    WHERE Home = ? OR Away = ?
    """
    cursor.execute(delete_statement, (team_name, team_name))
    conn.commit()
    conn.close()

    print(f"Matches involving '{team_name}' have been removed from the database.")

#remove_matches_involving_team ('d1_games.db', 'Grand View')
#update_team_names('d1_games.db', "Mount St. Mary’s", 'Saint Mary\'s')

def updateDatabase (df, db_path, table_name):
    # Write DataFrame to SQLite database
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
class DataAnalyzer:
    def __init__(self, games_df, teams_df, ratings_df, rating_prog_df):
        self.games_df = games_df
        self.teams_df = teams_df
        self.ratings_df = ratings_df
        # self.glicko_prog_df = glicko_prog_df
        self.rating_prog_df = rating_prog_df

    def printTeamSchedule(self, team_name, year=None):
        modified_games_df = self.games_df
        modified_games_df['new Date'] = pd.to_datetime(games_df['Date'])
        if year is not None:
            modified_games_df = modified_games_df[modified_games_df['new Date'].dt.year == year]
        printTeamSchedule (team_name, modified_games_df)

    def plotTeams(self, team_names: object, team_colors: object = None, year: object = None) -> object:
        plotTeams (team_names, self.rating_prog_df, team_colors, year)

    def simMatch (self, month, day, away_team, home_team, neutral_site=None):
            simMatch (month, day, away_team, home_team, self.rating_prog_df, neutral_site)

    def writeRatingsToExcelFile (self):
        file_name = 'TeamRatings.xlsx'
        self.ratings_df.to_excel(file_name, index=False, engine='openpyxl')

    def writeProgressionsToExcelFile (self):
        file_name = 'RatingProgression.xlsx'
        self.rating_prog_df.to_excel(file_name, index=False, engine='openpyxl')

    def writeEverythingToExcelFile (self):
        file_name = 'Test_games.xlsx'
        self.games_df.to_excel(file_name, index=False, engine='openpyxl')
        file_name = 'Test_prog_games.xlsx'
        self.rating_prog_df.to_excel(file_name, index=False, engine='openpyxl')

games_df, teams_df, ratings_df, elo_df, glicko_prog_df, elo_prog_df = set_up ()
autoCalc = DataAnalyzer (games_df, teams_df, elo_df, elo_prog_df)

# Note: if you want to calculate using glicko, uncomment the line below this
# and use that instead. It will, um, rerun set_up () though, so keep that in mind
# autoCalc_glicko = DataAnalyzer (games_df, teams_df, ratings_df, glicko_prog_df)

# autoCalc_plotTeams (['Georgia Tech', 'Virginia', 'Pittsburgh'])
#autoCalc_plotTeams (['LSU', 'Wake Forest'], [(70,29,124), (158, 126, 56)])
#autoCalc_plotTeams (['Virginia'])
# autoCalc_printTeamSchedule ('LSU')
# autoCalc_printTeamSchedule('Virginia')
# autoCalc_printTeamSchedule ('Wake Forest')
# autoCalc_printTeamSchedule ('Virginia')
# autoCalc_simMatch (6, 23, 'Wake Forest', 'LSU')

#autoCalc.simMatch (4,28,"LSU", "Florida", 0)
update_team_names('Data/ncaa_games_since_2000', 'Houston Baptist', 'Houston Christian')
autoCalc.printTeamSchedule ('Virginia', 2023)
autoCalc.printTeamSchedule ('Virginia')
autoCalc.printTeamSchedule ('Wake Forest')
autoCalc.printTeamSchedule ('Alcorn State')
autoCalc.simMatch ( 6, 26, 'Virginia', 'Virginia Tech')
#autoCalc.plotTeams (['Wake Forest', 'Florida', 'LSU', 'Vanderbilt', 'Virginia', 'Stanford', 'Miami', 'Tennessee', 'TCU', 'Oral Roberts'])
#autoCalc.plotTeams (['Virginia', 'Virginia Tech'], None, 2023)
autoCalc.plotTeams(['Virginia', 'Virginia Tech', 'Louisiana State', 'Clemson', 'Florida', 'Wake Forest', 'Texas Christian'], None,2023)
autoCalc.plotTeams (['Stanford', 'Tennessee', 'Louisiana State', 'Oral Roberts', 'Florida', 'Wake Forest',])
autoCalc.plotTeams (['Alcorn State', 'Virginia'])
autoCalc.writeRatingsToExcelFile()
autoCalc.writeProgressionsToExcelFile()
autoCalc.writeEverythingToExcelFile()

#Auto Calc functions:
    # printTeamSchedule (team_name, year (optional))
        # prints summary of games (results, elo results)
        # defaults to printing all games form last five years.
    # plotTeam (team_names, colors (optional))
        # plots elo over time
        # note both inputs are arrays. Prints everything from last 5 years (was originally only processing 1 year, haven't gotten around to updating plot yet)
    # simMatch (month, day, away_team, home_team, neutral_site (optional))
        # gives probability of a team winning on a date given the venue
        # input neutral_site = 1 if neutral
        # similar to plot teams, only handles 2023 dates for now
    # use: eg. autoCalc.printTeamSchedule ('Virginia').
    # Names have to be exact, so ctrl+f the .db file if you're unsure
#autoCalc.plotTeams(['Virginia', 'Virginia Tech', 'Louisiana State', 'Clemson', 'Florida', 'Wake Forest', 'Texas Christian'], None,2022)
