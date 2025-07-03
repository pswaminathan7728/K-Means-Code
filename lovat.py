import pandas as pd
import numpy as np

# ANSI escape codes for colors
COLOR_DARK_GREEN = "\033[32m"   # top 8
COLOR_LIME_GREEN = "\033[92m"   # next 8
COLOR_ORANGE = "\033[33m"       # 8 above bottom
COLOR_RED = "\033[31m"          # bottom 8
COLOR_YELLOW = "\033[93m"       # middle
COLOR_RESET = "\033[0m"

TELEOP_SECONDS = 135  # Teleop period in seconds

# Load CSV data (assumes file is named "lovatdata.csv")
df = pd.read_csv("lovatdata.csv")

# Convert teamNumber to string for uniformity
df['teamNumber'] = df['teamNumber'].astype(str)

# Define the columns that should be numeric for aggregation
numeric_cols = [
    "coralPickup", "algaePickup", "algaeKnocking", "underShallowCage", "teleopPoints",
    "autoPoints", "driverAbility", "feeds", "defends", "coralPickups", "algaePickups",
    "coralDrops", "algaeDrops", "coralL1", "coralL2", "coralL3", "coralL4", "processorScores",
    "netScores", "netFails", "activeAuton", "endgame"
]

# Ensure numeric columns are properly converted to numbers; convert errors to NaN, then fill with 0.
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(0)

# Aggregate data by teamNumber (summing numeric stats; for non-numeric, take the first)
agg_dict = {col: "sum" for col in numeric_cols}
agg_dict["role"] = "first"  # initial role from data (if provided)

team_stats = df.groupby("teamNumber", as_index=False).agg(agg_dict)

# Compute cycles and average teleop cycle time:
# Define cycles as the sum of scoring events: coralL1 + coralL2 + coralL3 + coralL4 + processorScores
team_stats["cycles"] = team_stats["coralL1"] + team_stats["coralL2"] + team_stats["coralL3"] + team_stats["coralL4"] + team_stats["processorScores"]
team_stats["avgCycleTime"] = team_stats["cycles"].apply(lambda x: TELEOP_SECONDS / x if x > 0 else np.nan)

# Compute role scores for role determination:
team_stats["coral_points"] = (
    team_stats["coralL1"] * 2 +
    team_stats["coralL2"] * 3 +
    team_stats["coralL3"] * 4 +
    team_stats["coralL4"] * 5 +
    team_stats["processorScores"] * 2
)
# Define algae_points (using provided and assumed weights)
team_stats["algae_points"] = (
    team_stats["algaePickup"] * 2 +
    team_stats["algaeKnocking"] * 3 -
    team_stats["algaeDrops"] * 2
)
# Define defense_points (using assumed weights)
team_stats["defense_points"] = (
    team_stats["defends"] * 3 +
    team_stats["feeds"] * 1 +
    team_stats["underShallowCage"] * 2
)

def determine_role(row):
    c = row["coral_points"]
    a = row["algae_points"]
    d = row["defense_points"]
    # Check if one metric is at least 20% higher than both others
    if c > a * 1.2 and c > d * 1.2:
        return "coral"
    elif a > c * 1.2 and a > d * 1.2:
        return "algae"
    elif d > c * 1.2 and d > a * 1.2:
        return "defense"
    else:
        return "all rounder"

team_stats["computed_role"] = team_stats.apply(determine_role, axis=1)

# For easy lookup later, create a dictionary mapping team number to its stats (row)
team_dict = {row["teamNumber"]: row for _, row in team_stats.iterrows()}

def color_for_rank(index, total):
    """
    Return the ANSI color code based on the ranking index.
    Index is zero-based.
    """
    # Top 8: dark green
    if index < 8:
        return COLOR_DARK_GREEN
    # Next 8: lime green
    elif index < 16:
        return COLOR_LIME_GREEN
    # For bottom 16 division, determine orange and red
    elif index >= total - 8:
        return COLOR_RED
    elif index >= total - 16:
        return COLOR_ORANGE
    else:
        return COLOR_YELLOW

def print_ranking(ranking_list, title):
    """
    ranking_list: list of tuples (teamNumber, value, computed_role)
    title: string title to display.
    """
    total = len(ranking_list)
    print(f"\n=== {title} ===")
    for idx, (team, value, role) in enumerate(ranking_list):
        color = color_for_rank(idx, total)
        print(f"{color}{idx+1:2d}. Team {team} [{role}] - {value}{COLOR_RESET}")

def rank_by_stat(stat):
    """
    Rank teams by an individual statistic column.
    """
    if stat not in team_stats.columns:
        print(f"Statistic '{stat}' not found.")
        return
    ranking = team_stats[['teamNumber', stat, "computed_role"]].copy()
    ranking = ranking.sort_values(by=stat, ascending=False)
    ranking_list = list(ranking.itertuples(index=False, name=None))
    print_ranking(ranking_list, f"Ranking by {stat}")

def rank_by_role(role):
    """
    Rank teams based on role-specific score. 
    For 'coral' ranking, include teams with computed_role in ['coral', 'all rounder'].
    Similarly for 'algae'. For 'defense', include only defense.
    For 'all rounder', include only all rounders.
    """
    role = role.lower()
    if role == "coral":
        subset = team_stats[team_stats["computed_role"].isin(["coral", "all rounder"])]
        metric = "coral_points"
    elif role == "algae":
        subset = team_stats[team_stats["computed_role"].isin(["algae", "all rounder"])]
        metric = "algae_points"
    elif role == "defense":
        subset = team_stats[team_stats["computed_role"] == "defense"]
        metric = "defense_points"
    elif role == "all rounder":
        subset = team_stats[team_stats["computed_role"] == "all rounder"]
        # For all rounders, use an average of the three scores
        subset = subset.copy()
        subset["avg_score"] = (subset["coral_points"] + subset["algae_points"] + subset["defense_points"]) / 3
        metric = "avg_score"
    else:
        print("Invalid role entered. Choose from: coral, algae, defense, all rounder")
        return

    ranking = subset[['teamNumber', metric, "computed_role"]].copy()
    ranking = ranking.sort_values(by=metric, ascending=False)
    ranking_list = list(ranking.itertuples(index=False, name=None))
    print_ranking(ranking_list, f"Ranking for role: {role}")

def display_team(team_number):
    """
    Display all aggregated statistics for a given team number.
    Also shows the team's rank (with color) in several key categories.
    """
    team_number = str(team_number)
    if team_number not in team_dict:
        print(f"Team {team_number} not found.")
        return
    team = team_dict[team_number]
    print(f"\n=== Statistics for Team {team_number} [{team['computed_role']}] ===")
    # Display all stats from team_stats row
    for col in team_stats.columns:
        if col == "teamNumber":
            continue
        print(f"{col}: {team[col]}")

    # Now, show ranking positions for a few key categories.
    key_stats = ["coralL4", "algaeDrops", "avgCycleTime", "coral_points", "algae_points", "defense_points"]
    for stat in key_stats:
        ranking = team_stats[['teamNumber', stat, "computed_role"]].copy().sort_values(by=stat, ascending=False)
        ranking = ranking.reset_index(drop=True)
        rank_idx = ranking[ranking["teamNumber"] == team_number].index[0]
        color = color_for_rank(rank_idx, len(ranking))
        print(f"{stat} Rank: {color}{rank_idx+1}{COLOR_RESET} out of {len(ranking)}")

def main_menu():
    while True:
        print("\nMenu Options:")
        print("1. Rank teams by individual statistic (e.g., coralL4, algaeDrops, etc.)")
        print("2. Rank teams by role (coral, algae, defense, all rounder)")
        print("3. Display a team's statistics (enter team number)")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            stat = input("Enter the statistic column name to rank by: ").strip()
            rank_by_stat(stat)
        elif choice == "2":
            role = input("Enter the role to rank by (coral, algae, defense, all rounder): ").strip()
            rank_by_role(role)
        elif choice == "3":
            team_number = input("Enter the team number: ").strip()
            display_team(team_number)
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main_menu()
