import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from adjustText import adjust_text  
from mpl_toolkits.mplot3d import Axes3D  

#  data preparation

# reading data pertaining to measured metrics and estimated metrics
team_stats = pd.read_csv("metrics.csv")
epa_stats = pd.read_csv("epa_metrics.csv")

# removing column spaces
team_stats.columns = team_stats.columns.str.strip()
epa_stats.columns = epa_stats.columns.str.strip()

team_stats["team"] = team_stats["team"].astype(str)
epa_stats["team"] = epa_stats["team"].astype(str)

# merging data based on the team 
all_the_data = pd.merge(team_stats, epa_stats, on="team", how="outer")

# filtering used metrics vs nonused metrics
raw_metrics_cols = [col for col in team_stats.columns if col not in ["num", "team"]]
epa_metric_cols = [col for col in epa_stats.columns if col not in ["num", "team", "first_event", "rank", "rps", "rps_per_match", "record"]]

# using ANSI Color Codes for organization and easy analysis

green_highlight = "\033[92m"
yellow_highlight = "\033[93m"
red_highlight = "\033[91m"
COLOR_RESET = "\033[0m"

def get_rank_color(current_rank, total_teams_count):
    """
    generates out a color code based on how good a team's rank is
    top 10 are green, bottom 10 are red , everyone else is yellow
    """
    if current_rank <= 10:
        return green_highlight
    elif current_rank >= total_teams_count - 9: # Checking if it's in the bottom 10.
        return red_highlight
    else:
        return yellow_highlight


def show_metric_rankings(metric_name):
    """
    Takes a metric, sorts all teams by it (best first), and then prints out a
    nicely colored list of who stands where. Super useful for quick comparisons!
    """
    if metric_name not in all_the_data.columns:
        print(f"\nHang on, I can't find a metric called '{metric_name}' in my data. Double-check the spelling?\n")
        return

    # Sorting everyone out from high to low for this specific metric.
    sorted_teams = all_the_data.sort_values(by=metric_name, ascending=False).reset_index(drop=True)
    how_many_teams = len(sorted_teams)

    print(f"\nHere's the leaderboard for '{metric_name}':")
    for i, team_info in sorted_teams.iterrows():
        this_teams_rank = i + 1
        rank_color = get_rank_color(this_teams_rank, how_many_teams)
        print(f"{rank_color}Rank {this_teams_rank}: Team {team_info['team']} with {metric_name} = {team_info[metric_name]}{COLOR_RESET}")


def get_team_info(team_id_input):
    """
     shows data metrics for a specific team
    """
 
    team_id_clean = str(team_id_input).strip()
    
    if team_id_clean not in all_the_data['team'].values:
        print(f"\nCan't find Team '{team_id_input}', is that the right team ID?\n")
        return

    one_team_data = all_the_data[all_the_data['team'] == team_id_clean].iloc[0]
    print(f"\nhere are the metrics for team {team_id_clean}:\n")
    
 
    all_known_metrics = raw_metrics_cols + epa_metric_cols
    total_teams_in_dataset = len(all_the_data)
    
    for each_metric in all_known_metrics:
        if each_metric in all_the_data.columns: 
            # Sorting the whole dataset by this metric to find our team's rank.
 
            sorted_by_this_metric = all_the_data.sort_values(by=each_metric, ascending=False)[each_metric]
            
            # Finding where our team's value for this metric sits in the sorted list.
      
            this_teams_rank = sorted_by_this_metric.reset_index(drop=True).tolist().index(one_team_data[each_metric]) + 1
            
        
            rank_display_color = get_rank_color(this_teams_rank, total_teams_in_dataset)
            
            # final print statement
            metric_value = one_team_data[each_metric]
            print(f"{each_metric}: {metric_value}  {rank_display_color}(Rank {this_teams_rank} of {total_teams_in_dataset}){COLOR_RESET}")
    print() # More breathing room.

# kmeans cluster analysis

def run_kmeans_analysis():
    """
   running a kmeans cluster analysis can help a lot with visualizing how good teams are at a combination of 2 or 3 metrics. graphing teams' statistics and clustering them in different groups can
   be very helpful in identifying green or red flag teams
    """
    # 2d or 3d analysis?
    how_many_dimensions = input("enter 2 or 3: ").strip()
    if how_many_dimensions not in ['2', '3']:
        print("pick either 2 or 3 please")
        return
    num_dims = int(how_many_dimensions)
    
    all_metrics_available = raw_metrics_cols + epa_metric_cols
    print("\nmetrics available for kmeans analysis:")
    for m in all_metrics_available:
        print(f" - {m}")
    
    # prompt user to choose metrics
    chosen_metrics = []
    for i in range(num_dims):
        chosen_metric = input(f"enter metric {i+1}: ").strip()
        if chosen_metric not in all_the_data.columns:
            print(f"'{chosen_metric}' is not in the list, please type it exactly")
            return
        chosen_metrics.append(chosen_metric)
    
    # asking user to provide a value for constant k
    try:
        num_clusters = int(input("how many clusters (groups)? ").strip())
    except ValueError:
        print("try again with a whole number")
        return
    
   
    data_for_clustering = all_the_data[['team'] + chosen_metrics].dropna().copy()
    
    # double check that metrics are not strings,but are numbers
    for chosen_m in chosen_metrics:
        data_for_clustering[chosen_m] = pd.to_numeric(data_for_clustering[chosen_m], errors='coerce')
    data_for_clustering = data_for_clustering.dropna() # dropping rows that become empty

    if data_for_clustering.empty:
        print("clustering not available, sorry")
        return

    
    k_means_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto') 
    data_for_clustering['cluster'] = k_means_model.fit_predict(data_for_clustering[chosen_metrics])
    
    # colors to distinguish different clusters
    cluster_colors = plt.get_cmap("colors", num_clusters)
    
    if num_dims == 2:
        plt.figure(figsize=(10, 8)) 
        all_the_labels = [] 
        for cluster_num in range(num_clusters):
            this_cluster_data = data_for_clustering[data_for_clustering['cluster'] == cluster_num]
            plt.scatter(this_cluster_data[chosen_metrics[0]], this_cluster_data[chosen_metrics[1]],  
                        label=f"cluster {cluster_num}",
                        color=cluster_colors(cluster_num))
            for _, team_row in this_cluster_data.iterrows():
                txt_label = plt.text(team_row[chosen_metrics[0]], team_row[chosen_metrics[1]], team_row['team'], fontsize=8, alpha=0.7)
                all_the_labels.append(txt_label)
        adjust_text(all_the_labels, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        plt.xlabel(chosen_metrics[0])
        plt.ylabel(chosen_metrics[1])
        plt.title(f"k-means clustering (k={num_clusters}) for {chosen_metrics[0]} vs {chosen_metrics[1]}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6) 
        plt.show() 
    elif num_dims == 3:
        fig = plt.figure(figsize=(12, 10)) 
        ax = fig.add_subplot(111, projection='3d') # setting up the 3D axes
        for cluster_num in range(num_clusters):
            this_cluster_data = data_for_clustering[data_for_clustering['cluster'] == cluster_num]
            ax.scatter(this_cluster_data[chosen_metrics[0]],  
                       this_cluster_data[chosen_metrics[1]],  
                       this_cluster_data[chosen_metrics[2]],
                       label=f"cluster {cluster_num}",
                       color=cluster_colors(cluster_num))
            for _, team_row in this_cluster_data.iterrows():
                ax.text(team_row[chosen_metrics[0]], team_row[chosen_metrics[1]], team_row[chosen_metrics[2]],  
                        team_row['team'], fontsize=8, alpha=0.8)
        
        # drawing a line in the (1,1,1) direction, helps me visualize direction and acts as a referenec for direction of the max of the 3 metrics
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()
        t_start = min(x_limits[0], y_limits[0], z_limits[0])
        t_end = max(x_limits[1], y_limits[1], z_limits[1])
        ax.plot([t_start, t_end], [t_start, t_end], [t_start, t_end], color='red', linewidth=2, linestyle=':', label='direction (1,1,1) Ref.')
        
        ax.set_xlabel(chosen_metrics[0])
        ax.set_ylabel(chosen_metrics[1])
        ax.set_zlabel(chosen_metrics[2])
        ax.set_title(f"3d k-means clustering (k={num_clusters}) on {', '.join(chosen_metrics)}")
        ax.legend()
        plt.show()

# main loop

def main():
    print("kmeans & ranking analysis")
    
    while True: 
        print("\nWhat do you want to do?")
        print("  1: teams ranked by a specific EPA metric")
        print("  2: full breakdown for a single team")
        print("  3: k-means clustering")
        print("  q: quit")
        
        user_choice = input("\ntype your choice (1, 2, 3, or q): ").strip().lower()
        
        if user_choice == '1':
            print("\nhere are the EPA metrics")
            for metric in epa_metric_cols:
                print(f" - {metric}")
            metric_to_rank = input("\nWhich metric , type it exactly: ").strip()
            show_metric_rankings(metric_to_rank)
        elif user_choice == '2':
            # remind users of metrics they can potentially choose
            print("\n these are all of the metrics for a team:")
            all_known_metrics_for_overview = raw_metrics_cols + epa_metric_cols
            for metric in all_known_metrics_for_overview:
                print(f" - {metric}")
            team_id_lookup = input("\nenter in a team number (e.g., 'team 123'): ").strip()
            get_team_info(team_id_lookup)
        elif user_choice == '3':
            run_kmeans_analysis()
        elif user_choice == 'q':
            print("quitting")
            break 
        else:
            print("please pick from one of the options")

if __name__ == "__main__":
    main()
