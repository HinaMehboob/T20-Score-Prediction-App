import pandas as pd # pyright: ignore 
import numpy as np # pyright: ignore 
import matplotlib.pyplot as plt # pyright: ignore 
import seaborn as sns # pyright: ignore
import pickle # pyright: ignore
import os
from sklearn.model_selection import train_test_split # pyright: ignore
from sklearn.ensemble import RandomForestRegressor # pyright: ignore
from sklearn.metrics import r2_score # pyright: ignore

# 1. SETUP
# Create a folder to save the images
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load Data
print("Loading data...")
df = pd.read_csv("data/t20_first_innings.csv")

# Filter and Feature Engineering (Same as training)
df = df[(df['final_score'] > 50) & (df['balls_remaining'] >= 0)]
df['balls_bowled'] = (df['over'] * 6) + df['ball']
df['run_rate'] = df['current_score'] / (df['over'] + df['ball']/6 + 1e-5)
df['wickets_remaining'] = 10 - df['wickets']
df['runs_last_5_normalized'] = df['runs_last_5_overs'] / 5

features = ['over', 'ball', 'current_score', 'wickets', 'balls_remaining', 
            'runs_last_5_overs', 'run_rate', 'wickets_remaining', 'runs_last_5_normalized']

X = df[features]
y = df['final_score']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Model
print("Loading model...")
with open('models/pipe.pkl', 'rb') as f:
    model = pickle.load(f)

# Make Predictions for Plotting
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Model R2 Score: {r2:.2f}")


# 2. GENERATE PLOTS

# PLOT 1: CORRELATION MATRIX (The "Matrix" you wanted) 
plt.figure(figsize=(10, 8))
corr_matrix = df[features + ['final_score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig('plots/1_correlation_matrix.png')
print("Saved: plots/1_correlation_matrix.png")

#  PLOT 2: ACTUAL VS PREDICTED (The standard accuracy plot) 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='#3498db', edgecolor='k', s=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=3, label='Perfect Prediction Line')
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title(f"Actual vs Predicted Scores (R² = {r2:.2f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/2_actual_vs_predicted.png')
print("Saved: plots/2_actual_vs_predicted.png")

# PLOT 3: FEATURE IMPORTANCE (What matters most?) 
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names, palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('plots/3_feature_importance.png')
print("Saved: plots/3_feature_importance.png")

# PLOT 4: RESIDUAL PLOT (Analysis of Errors) 
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.3, color='#e74c3c', s=20)
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.xlabel("Predicted Scores")
plt.ylabel("Residuals (Error)")
plt.title("Residual Plot (Checking for Patterns in Errors)")
plt.grid(True, alpha=0.3)
plt.savefig('plots/4_residual_plot.png')
print("Saved: plots/4_residual_plot.png")

# PLOT 5: ERROR DISTRIBUTION (Histogram) 
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='#2ecc71')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error (Runs)")
plt.ylabel("Frequency")
plt.axvline(x=0, color='k', linestyle='--')
plt.savefig('plots/5_error_distribution.png')
print("Saved: plots/5_error_distribution.png")

print("\nAll plots generated in the 'plots/' folder!")