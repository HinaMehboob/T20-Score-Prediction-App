import yaml # pyright: ignore
import os
import pandas as pd # pyright: ignore


T20_YAML_FOLDER = "t20s"          # folder containing YAML files
OUTPUT_CSV = "t20_first_innings.csv"


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


data_rows = []

files = os.listdir(T20_YAML_FOLDER)
print(f"Found {len(files)} files")

for file_name in files:
    if not file_name.endswith(".yaml"):
        continue

    file_path = os.path.join(T20_YAML_FOLDER, file_name)

    try:
        match_data = load_yaml(file_path)

        innings = match_data["innings"]
        first_innings = innings[0]
        innings_name = list(first_innings.keys())[0]
        innings_data = first_innings[innings_name]

        deliveries = innings_data["deliveries"]

        # First pass: get final score
        final_score = 0
        for delivery in deliveries:
            ball = list(delivery.keys())[0]
            info = delivery[ball]
            final_score += info["runs"]["total"]

        # Second pass: feature engineering 
        current_score = 0
        wickets = 0
        recent_runs = []
        ball_count = 0

        for delivery in deliveries:
            ball = list(delivery.keys())[0]
            info = delivery[ball]

            over, ball_no = map(int, str(ball).split("."))
            runs = info["runs"]["total"]

            current_score += runs
            ball_count += 1
            recent_runs.append(runs)

            if "wickets" in info:
                wickets += len(info["wickets"])

            if len(recent_runs) > 30:   # last 5 overs
                recent_runs.pop(0)

            balls_remaining = 120 - ball_count

            data_rows.append({
                "over": over,
                "ball": ball_no,
                "current_score": current_score,
                "wickets": wickets,
                "balls_remaining": balls_remaining,
                "runs_last_5_overs": sum(recent_runs),
                "final_score": final_score
            })

    except Exception as e:
        # Skip corrupted / incomplete matches
        continue


df = pd.DataFrame(data_rows)

print("Final dataset shape:", df.shape)
print(df.head())

df.to_csv(OUTPUT_CSV, index=False)
print(f"Dataset saved as {OUTPUT_CSV}")
