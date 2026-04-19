# ⚡️ AEnergy: Precision AI Energy Platform

AEnergy is a modern, high-performance web application designed to predict monthly electricity costs of home appliances with professional accuracy. It leverages a robust Machine Learning engine and a premium "Holographic Glassmorphism" interface to provide actionable energy insights for both residential and B2B users.

---

## ✨ Key Features

- **🧠 Intelligent Cost Estimator**: Powered by a **Random Forest Regressor** that captures non-linear relationships between power wattage, usage hours, and energy efficiency ratings.
- **🎨 Premium UX/UI**: A customized "Holographic Glassmorphism" light mode featuring irisdescending gradients, frosted glass effects, and smooth micro-animations.
- **📊 Real-time Insights**: Interactive Plotly visualizations including:
    - **Smart Gauge**: Real-time comparison between AI prediction and math-based standard rates.
    - **Trend Analysis**: Scatter and Bar charts analyzing the correlation between consumption (kWh) and appliance efficiency labels.
- **📝 Data Collector**: Built-in specialized form to seed real-world bill data for continuous AI model training.
- **🔄 Engine Control**: One-click model retraining directly from the UI with real-time accuracy scoring (R²).

---

## 🛠 Tech Stack

- **Backend Logic**: Python 3.9+ 
- **AI Engine**: Scikit-Learn (RandomForestRegressor), Joblib
- **Frontend**: Streamlit (with Custom CSS Injection)
- **Visualization**: Plotly Express & Graph Objects
- **Data Management**: Pandas & CSV-based storage

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher installed on your system.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CHANXYII/AEnergy-app.git
   cd AEnergy-app
   ```

2. Create and activate a Virtual Environment (Recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Start the Streamlit server directly:
```bash
streamlit run front-end/app.py
```
Then, visit `http://localhost:8501` in your browser.

---

## 📂 Project Structure

- `front-end/app.py`: Main application UI and styling logic.
- `front-end/backend.py`: AI model management, training, and data processing.
- `collected_data.csv`: Historical dataset for training.
- `model.pkl`: Serialized Random Forest model.
- `requirements.txt`: Project dependencies.

---

## ⚖️ License
© 2026 AEnergy Platform | B2B Precision AI Engine
