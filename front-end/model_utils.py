import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def train_and_save_model(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) < 10:
        return None, "ข้อมูลยังน้อยเกินไป (ต้องการอย่างน้อย 10 samples)"

    X = df[['wattage', 'hours', 'days', 'efficiency']]
    y = df['cost']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return {"r2": r2, "mse": mse}, None
