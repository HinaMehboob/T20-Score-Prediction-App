import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np  # type: ignore
import pickle
import os
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestRegressor # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # pyright: ignore[reportMissingModuleSource]


# 1. SETUP PATHS

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 't20_first_innings.csv')
model_save_path = os.path.join(base_dir, 'models', 'pipe.pkl')


# 2. LOAD DATA

print(f"Loading data from: {data_path}")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print("Error: Could not find the CSV file. Please check if 't20_first_innings.csv' is inside the 'data' folder.")
    exit()

# Filter valid data
df = df[(df['final_score'] > 50) & (df['balls_remaining'] >= 0)]


# 3. FEATURE ENGINEERING

# Calculate Run Rate
df['balls_bowled'] = (df['over'] * 6) + df['ball']
df['run_rate'] = df['current_score'] / (df['over'] + df['ball']/6 + 1e-5)

# Calculate Wickets Left
df['wickets_remaining'] = 10 - df['wickets']

# Normalize Runs in Last 5 Overs
df['runs_last_5_normalized'] = df['runs_last_5_overs'] / 5

# Define Features
features = ['over', 'ball', 'current_score', 'wickets', 'balls_remaining', 
            'runs_last_5_overs', 'run_rate', 'wickets_remaining', 'runs_last_5_normalized']

X = df[features]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. MODEL TRAINING & COMPARISON

# A. Linear Regression (Baseline)
print("\n--- Training Linear Regression (Baseline) ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression MAE : {mae_lr:.2f}")
print(f"Linear Regression R²  : {r2_lr:.4f}")

# B. Random Forest (Champion Model)
print("\n--- Training Random Forest (Champion) ---")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MAE     : {mae_rf:.2f}")
print(f"Random Forest R²      : {r2_rf:.4f}")

#  5. SAVE MODEL

# We save the Random Forest model because it is the "Best Performer"
print(f"\nSaving best model (Random Forest) to: {model_save_path}")
with open(model_save_path, 'wb') as f:
    pickle.dump(rf_model, f)

print("Success! Model saved.")