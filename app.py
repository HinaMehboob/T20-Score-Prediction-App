from flask import Flask, render_template, request # pyright: ignore
import pickle
import numpy as np # pyright: ignore
import os

app = Flask(__name__)

# 1. Load the Trained Model
# We use os.path to safely find the model file in the 'models' folder
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'pipe.pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Run 'train_model.py' first.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 2. Get Raw Input from HTML Form
            overs_input = float(request.form['overs'])   # e.g. 10.2
            current_score = int(request.form['score'])
            wickets = int(request.form['wickets'])
            runs_last_5 = int(request.form['runs_last_5'])

            # 3. Feature Engineering (MUST Match 'train_model.py' exactly)
            
            # Extract Over and Ball
            over = int(overs_input)
            ball = int(round((overs_input - over) * 10))
            
            # Calculate Derived Features
            balls_bowled = (over * 6) + ball
            balls_remaining = 120 - balls_bowled
            wickets_remaining = 10 - wickets
            
            # Calculate Run Rate (avoid division by zero)
            if balls_bowled == 0:
                run_rate = 0
            else:
                run_rate = current_score / (over + ball/6)
            
            # Normalize recent runs
            runs_last_5_normalized = runs_last_5 / 5

            # 4. Create Input Array for Model
            # Columns: ['over', 'ball', 'current_score', 'wickets', 'balls_remaining', 
            #           'runs_last_5_overs', 'run_rate', 'wickets_remaining', 'runs_last_5_normalized']
            
            input_features = np.array([[
                over, 
                ball, 
                current_score, 
                wickets, 
                balls_remaining, 
                runs_last_5, 
                run_rate, 
                wickets_remaining, 
                runs_last_5_normalized
            ]])

            # 5. Make Prediction
            prediction = model.predict(input_features)[0]
            final_score = int(prediction)

            return render_template('index.html', prediction=final_score)

        except Exception as e:
            return render_template('index.html', error=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)