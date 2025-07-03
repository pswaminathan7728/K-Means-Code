import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from adjustText import adjust_text  # for adjusting text annotations in 2D
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# ----------------- Data Loading and Preparation -----------------

# Load the CSV files (adjust the file paths if needed)
df_metrics = pd.read_csv("metrics.csv")
df_epa = pd.read_csv("epa_metrics.csv")

# Clean column names: strip whitespace.
df_metrics.columns = df_metrics.columns.str.strip()
df_epa.columns = df_epa.columns.str.strip()

# Convert the 'team' column to string for consistency.
df_metrics["team"] = df_metrics["team"].astype(str)
df_epa["team"] = df_epa["team"].astype(str)

# Merge the two DataFrames on the 'team' column.
df = pd.merge(df_metrics, df_epa, on="team", how="outer")

# Define metric lists based on the CSV structures.
# For df_metrics, ignore 'num' and 'team'.
metrics_columns = [col for col in df_metrics.columns if col not in ["num", "team"]]

# For df_epa, ignore non-metric columns.
epa_columns = [col for col in df_epa.columns if col not in ["num", "team", "first_event", "rank", "rps", "rps_per_match", "record"]]

# ----------------- ANSI Color Codes (for terminal output) -----------------

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def get_color(rank, total):
    """
    Return a color code based on the team's rank.
    Top 10: Green, Bottom 10: Red, Otherwise: Yellow.
    """
    if rank <= 10:
        return GREEN
    elif rank >= total - 9:
        return RED
    else:
        return YELLOW

# ----------------- Option 1: Ranking by Metric -----------------

def rank_by_metric(metric):
    """
    Sort teams (highest-to-lowest) by the given metric and print rankings.
    """
    if metric not in df.columns:
        print(f"\nMetric '{metric}' not found in the data.\n")
        return
    sorted_df = df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    total_teams = len(sorted_df)
    print(f"\nRanking for metric '{metric}':")
    for i, row in sorted_df.iterrows():
        rank = i + 1
        color = get_color(rank, total_teams)
        print(f"{color}Rank {rank}: Team {row['team']} with {metric} = {row[metric]}{RESET}")
    print()  # newline for spacing

# ----------------- Option 2: Team Overview -----------------

def team_overview(team_input):
    """
    Display all metrics (from both CSVs) for a given team along with its ranking in each metric.
    """
    team_val = str(team_input).strip()
    
    if team_val not in df['team'].values:
        print(f"\nTeam '{team_input}' not found in data.\n")
        return

    team_row = df[df['team'] == team_val].iloc[0]
    print(f"\nOverview for Team {team_val}:\n")
    
    # Combine metric lists from both CSVs.
    all_metrics = metrics_columns + epa_columns
    total_teams = len(df)
    
    for metric in all_metrics:
        if metric in df.columns:
            # Sort teams by the metric and compute the rank for this team.
            sorted_series = df.sort_values(by=metric, ascending=False)[metric]
            rank = sorted_series.reset_index(drop=True).tolist().index(team_row[metric]) + 1
            color = get_color(rank, total_teams)
            value = team_row[metric]
            print(f"{metric}: {value}  {color}(Rank {rank} of {total_teams}){RESET}")
    print()  # newline for spacing

# ----------------- Option 3: K-Means Clustering Analysis -----------------

def kmeans_clustering():
    """
    Perform K-Means clustering on either 2 or 3 chosen metrics and graph the results.
    """
    # Ask for analysis dimension.
    dimension = input("Enter analysis dimension").strip()
    if dimension not in ['2', '3']:
        print("Invalid dimension. Please enter 2 or 3.")
        return
    dim = int(dimension)
    
    # Prepare a combined list of metrics.
    all_metrics = metrics_columns + epa_columns
    print("\nAvailable Metrics for K-Means Clustering:")
    for metric in all_metrics:
        print(f" - {metric}")
    
    # Ask for metric names based on desired dimension.
    selected_metrics = []
    for i in range(dim):
        metric = input(f"Enter metric {i+1}: ").strip()
        if metric not in df.columns:
            print(f"Metric '{metric}' not found. Please try again.")
            return
        selected_metrics.append(metric)
    
    # Ask for the number of clusters.
    try:
        k = int(input("Enter number of clusters (k): ").strip())
    except ValueError:
        print("Invalid input for number of clusters.")
        return
    
    # Extract the team names along with the selected metrics and drop rows with missing values.
    clustering_data = df[['team'] + selected_metrics].dropna().copy()
    # Convert metric values to numeric (if they aren't already)
    for metric in selected_metrics:
        clustering_data[metric] = pd.to_numeric(clustering_data[metric], errors='coerce')
    clustering_data = clustering_data.dropna()

    if clustering_data.empty:
        print("No valid data available for these metrics.")
        return

    # Run K-Means clustering.
    kmeans = KMeans(n_clusters=k, random_state=42)
    clustering_data['cluster'] = kmeans.fit_predict(clustering_data[selected_metrics])
    
    # Colors for clusters.
    colors = plt.get_cmap("viridis", k)
    
    if dim == 2:
        # 2D Plotting.
        plt.figure(figsize=(8, 6))
        texts = []  # store text annotations for adjustText.
        
        for cluster_label in range(k):
            cluster_data = clustering_data[clustering_data['cluster'] == cluster_label]
            plt.scatter(cluster_data[selected_metrics[0]], cluster_data[selected_metrics[1]], 
                        label=f"Cluster {cluster_label}",
                        color=colors(cluster_label))
            # Annotate each point with the team name.
            for _, row in cluster_data.iterrows():
                txt = plt.text(row[selected_metrics[0]], row[selected_metrics[1]], row['team'], fontsize=8, alpha=0.7)
                texts.append(txt)
        
        # Adjust text annotations to reduce overlapping.
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
        plt.xlabel(selected_metrics[0])
        plt.ylabel(selected_metrics[1])
        plt.title(f"K-Means Clustering (k={k}) on {selected_metrics[0]} vs {selected_metrics[1]}")
        plt.legend()
        plt.show()
        
    elif dim == 3:
        # 3D Plotting.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_label in range(k):
            cluster_data = clustering_data[clustering_data['cluster'] == cluster_label]
            ax.scatter(cluster_data[selected_metrics[0]], 
                       cluster_data[selected_metrics[1]], 
                       cluster_data[selected_metrics[2]],
                       label=f"Cluster {cluster_label}",
                       color=colors(cluster_label))
            # Annotate each point with the team name.
            for _, row in cluster_data.iterrows():
                ax.text(row[selected_metrics[0]], row[selected_metrics[1]], row[selected_metrics[2]], 
                        row['team'], fontsize=8, alpha=0.8)
        
        # Draw the vector in the direction (1,1,1) that spans the limits of the plot.
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        zlims = ax.get_zlim()
        t_min = min(xlims[0], ylims[0], zlims[0])
        t_max = max(xlims[1], ylims[1], zlims[1])
        ax.plot([t_min, t_max], [t_min, t_max], [t_min, t_max], color='red', linewidth=2, label='Direction (1,1,1)')
        
        ax.set_xlabel(selected_metrics[0])
        ax.set_ylabel(selected_metrics[1])
        ax.set_zlabel(selected_metrics[2])
        ax.set_title(f"3D K-Means Clustering (k={k}) on {', '.join(selected_metrics)}")
        ax.legend()
        plt.show()

# ----------------- Main Loop -----------------

def main():
    print("Welcome to WarriorBenchmark!")
    
    while True:
        print("\nChoose an option:")
        print("  1: Rank teams by an EPA metric")
        print("  2: Show a team's overview (all metrics)")
        print("  3: K-Means clustering analysis")
        print("  q: Quit")
        
        choice = input("\nEnter your choice (1/2/3/q): ").strip().lower()
        
        if choice == '1':
            print("\nAvailable EPA Metrics:")
            for metric in epa_columns:
                print(f" - {metric}")
            metric = input("\nEnter the metric name from the list above: ").strip()
            rank_by_metric(metric)
        elif choice == '2':
            print("\nAvailable Metrics for Team Overview:")
            all_metrics = metrics_columns + epa_columns
            for metric in all_metrics:
                print(f" - {metric}")
            team_input = input("\nEnter the team identifier: ").strip()
            team_overview(team_input)
        elif choice == '3':
            kmeans_clustering()
        elif choice == 'q':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
