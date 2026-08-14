import pandas as pd
import json
import os

def main():
    TRAIN_PATH = "data/train.csv"
    OUT_PATH = "data/pitcher_final_stats.json"
    
    print(f"Loading data from {TRAIN_PATH}...")
    # Load only necessary columns for speed
    df = pd.read_csv(TRAIN_PATH, usecols=[
        'pitcher_id', 'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate'
    ])
    
    print("Extracting the last recorded stats for each pitcher...")
    # Since the data is chronologically sorted, keeping the 'last' occurrence gives the most recent stats
    last_stats = df.drop_duplicates(subset=['pitcher_id'], keep='last')
    
    # Convert to a dictionary format: { pitcher_id: { stats... } }
    # Using string for pitcher_id to ensure JSON compatibility
    stats_dict = {}
    for _, row in last_stats.iterrows():
        pid = str(row['pitcher_id'])
        stats_dict[pid] = {
            "asof_pitcher_success_rate": row['asof_pitcher_success_rate'],
            "asof_pitcher_reverse_rate": row['asof_pitcher_reverse_rate']
        }
        
    print(f"Saving extracted stats for {len(stats_dict)} pitchers to {OUT_PATH}...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=4)
        
    print("[SUCCESS] Done!")

if __name__ == "__main__":
    main()
