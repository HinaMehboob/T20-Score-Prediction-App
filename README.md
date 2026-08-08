# T20 Cricket Score Predictor 🏏
Welcome to the **T20 Cricket Score Predictor** project! This is an end-to-end Machine Learning pipeline that predicts the final score of a T20 cricket match's first innings based on the current match situation.
The project encompasses data extraction, feature engineering, model training, evaluation plotting, and a Flask-based web application to serve the predictions.
## 📌 Features
- **Data Parsing:** Extracts ball-by-ball T20 match data from YAML files.
- **Feature Engineering:** Calculates crucial cricket metrics like run rate, balls remaining, wickets left, and runs scored in the last 5 overs.
- **Model Training:** Compares a baseline Linear Regression model against a champion Random Forest Regressor to achieve high accuracy.
- **Data Visualization:** Generates comprehensive evaluation plots (Correlation Matrix, Feature Importance, Residuals, etc.) to interpret the model's behavior.
- **Web Interface:** A sleek and simple Flask web application that allows users to input the current match situation and get an instant predicted final score.
## 📂 Project Structure
```text
cricket_ml/
│
├── app.py                     # Flask web application for serving the predictor
├── extract_t20_features.py    # Script to parse YAML files and generate the CSV dataset
├── train_model.py             # Script to train the ML models and save the best one (pipe.pkl)
├── generate_plots.py          # Script to generate model evaluation visualizations
│
├── data/                      # Directory for data (e.g., t20_first_innings.csv)
├── models/                    # Directory where the trained model (pipe.pkl) is saved
├── plots/                     # Directory for generated evaluation plots
├── templates/                 # HTML templates for the Flask app (e.g., index.html)
└── notebooks/                 # Directory for exploratory Jupyter notebooks
```
## 🚀 Getting Started
### 1. Prerequisites
Ensure you have Python installed. The project relies on the following major libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `PyYAML`
- `Flask`

Install the required packages using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml Flask
```
### 2. Prepare the Data
If you have raw T20 YAML files, place them in a folder named `t20s/` in the root directory. Then, run the extraction script to generate the dataset (`data/t20_first_innings.csv`):
```bash
python extract_t20_features.py
```
*(Note: If you already have `t20_first_innings.csv` in your `data/` folder, you can skip this step.)*
### 3. Train the Model
Run the training script to process the CSV data, train the Linear Regression and Random Forest models, and save the best performer (`models/pipe.pkl`):
```bash
python train_model.py
```
This script will output the Mean Absolute Error (MAE) and R² score for both models.
### 4. Generate Visualizations (Optional)
To understand how the model is making decisions and analyze its errors, run the plotting script:
```bash
python generate_plots.py
```
This will generate several images in the `plots/` folder:
- `1_correlation_matrix.png`
- `2_actual_vs_predicted.png`
- `3_feature_importance.png`
- `4_residual_plot.png`
- `5_error_distribution.png`
### 5. Run the Web Application
Start the Flask server to interact with the predictor via a web interface:
```bash
python app.py
```
Open your web browser and navigate to `http://127.0.0.1:5000/`. Enter the current match statistics to get the predicted final score!
---
## 🧠 How it Works (Under the Hood)
The predictor uses the following features to make its prediction:
1. **Overs Bowled:** (`over` & `ball`)
2. **Current Score:** Total runs scored so far.
3. **Wickets Fallen:** Total wickets lost.
4. **Balls Remaining:** `120 - balls_bowled`
5. **Wickets Remaining:** `10 - wickets`
6. **Current Run Rate:** `current_score / overs_bowled`
7. **Runs in Last 5 Overs:** Captures the current momentum of the batting team.
8. **Normalized Last 5 Overs:** `runs_last_5_overs / 5`
The **Random Forest Regressor** proved to be the most effective model for this task, as it can capture non-linear relationships (e.g., the compounding effect of having wickets in hand during the final overs).
---
## 📝 Future Improvements
- Incorporate ground-specific data (pitch conditions, stadium dimensions).
- Add team and player-specific statistics (batsman strike rates, bowler economy rates).
- Extend to predict 2nd innings run chases and win probabilities.
