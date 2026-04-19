import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

CSV_FILE = 'collected_data.csv'
MODEL_FILE = 'model.pkl'

def load_data():
    if not os.path.exists(CSV_FILE):
        # Generate baseline dataset so it's instantly usable!
        np.random.seed(42)
        n_samples = 50
        wattages = np.random.choice([10, 50, 100, 500, 1000, 2000, 3000], n_samples)
        hours = np.random.uniform(0.5, 24, n_samples)
        days = np.random.randint(1, 32, n_samples)
        efficiencies = np.random.choice([1, 2, 3, 4, 5], n_samples)
        
        costs = []
        for w, h, d, e in zip(wattages, hours, days, efficiencies):
            kwh = (w * h * d) / 1000
            # Efficiency non-linearly affects cost: higher e means lower cost
            efficiency_discount = 1.0 - (e * 0.05)
            c = kwh * 4.5 * efficiency_discount
            # Add some statistical noise
            c = c + np.random.normal(0, max(1, c * 0.05))
            costs.append(round(max(0, c), 2))
            
        df = pd.DataFrame({
            'wattage': wattages,
            'hours': np.round(hours, 1),
            'days': days,
            'efficiency': efficiencies,
            'cost': costs
        })
        df.to_csv(CSV_FILE, index=False)
    return pd.read_csv(CSV_FILE)

def calculate_kwh(w, h, d):
    return (w * h * d) / 1000.0

def train_and_save_model(csv_path=CSV_FILE):
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 5: return None, "Insufficient data (minimum 5 rows needed for random forest)"
        
        # Feature Engineering
        df['kwh'] = df.apply(lambda row: calculate_kwh(row['wattage'], row['hours'], row['days']), axis=1)
        
        X = df[['wattage', 'hours', 'days', 'efficiency', 'kwh']]
        y = df['cost']
        
        # Replace LinearRegression with RandomForest to capture nonlinear efficiency dynamics
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
            
        return {"r2": r2_score(y, model.predict(X))}, None
    except Exception as e:
        return None, str(e)

def add_record(w, h, d, e, cost):
    # Load and save new record
    new_row = pd.DataFrame([{
        'wattage': w, 'hours': h, 'days': d, 'efficiency': e, 'cost': cost
    }])
    new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
    return True

def predict_cost(p_w, p_h, p_d, p_e):
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, 'rb') as f: 
        model = pickle.load(f)
        
    kwh = calculate_kwh(p_w, p_h, p_d)
    input_data = pd.DataFrame([{
        'wattage': p_w, 
        'hours': p_h, 
        'days': p_d, 
        'efficiency': p_e,
        'kwh': kwh
    }])
    
    pred = model.predict(input_data)[0]
    return max(0, pred)
