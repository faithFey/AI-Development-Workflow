
# ======================
# 1. Imports & Setup
# ======================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ======================
# 2. Synthetic Data Generation
# ======================
def generate_synthetic_data(years=10):
    np.random.seed(42)
    dates = pd.date_range(start='2013-01-01', periods=years*12, freq='M')
    data = {
        'Date': dates,
        'Maize_Production_MT': np.random.normal(1000, 200, len(dates)).cumsum(),
        'Rainfall_mm': np.random.gamma(shape=2, scale=30, size=len(dates)),
        'Drought_Index': np.random.uniform(0, 0.5, len(dates)),
        'Price_KES': np.random.normal(2500, 300, len(dates)).cumsum()
    }
    df = pd.DataFrame(data)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

df = generate_synthetic_data()
print(df.head())

# ======================
# 3. Data Preprocessing
# ======================
# Handle missing values
df.fillna(method='ffill', inplace=True)

# Normalization
scaler = MinMaxScaler()
features = ['Maize_Production_MT', 'Rainfall_mm', 'Drought_Index', 'Price_KES']
df[features] = scaler.fit_transform(df[features])

# Time-series formatting
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 0])  # Predicting production
    return np.array(X), np.array(y)

n_steps = 12  # 1-year lookback
X, y = create_sequences(df[features].values, n_steps)

# Train/Val/Test Split (temporal)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# ======================
# 4. Model Development
# ======================
model = Sequential([
    LSTM(64, activation='relu', input_shape=(n_steps, len(features))),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ======================
# 5. Evaluation
# ======================
def evaluate_model(model, X, y):
    loss, mae = model.evaluate(X, y, verbose=0)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {np.sqrt(loss):.4f}")

print("\nTest Set Evaluation:")
evaluate_model(model, X_test, y_test)

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Training History')
plt.show()

# ======================
# 6. Concept Drift Monitoring
# ======================
class ConceptDriftDetector:
    def __init__(self, window_size=6, threshold=0.15):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_rmse = None

    def update_baseline(self, model, X_val, y_val):
        loss = model.evaluate(X_val, y_val, verbose=0)[0]
        self.baseline_rmse = np.sqrt(loss)

    def check_drift(self, model, X_new, y_new):
        current_loss = model.evaluate(X_new, y_new, verbose=0)[0]
        current_rmse = np.sqrt(current_loss)
        drift = (current_rmse - self.baseline_rmse) / self.baseline_rmse
        return drift > self.threshold

# Initialize detector
drift_detector = ConceptDriftDetector()
drift_detector.update_baseline(model, X_val, y_val)

# Simulate drift detection
print(f"\nDrift detected: {drift_detector.check_drift(model, X_test, y_test)}")

# ======================
# 7. Deployment Simulation
# ======================
def predict_shortage(model, latest_data, shortage_threshold=0.3):
    """Predict if shortage will occur next month"""
    prediction = model.predict(latest_data.reshape(1, n_steps, len(features)))
    return prediction[0][0] < shortage_threshold

# Example usage
latest_sequence = X_test[-1]  # Most recent data
if predict_shortage(model, latest_sequence):
    print("\nALERT: Potential grain shortage predicted next month!")
else:
    print("\nNo shortage predicted")


# In[7]:


# Ensure your notebook is saved with the correct name
from google.colab import drive
drive.mount('/content/drive')  # Only if using Google Drive

# Save a copy (replace with your actual notebook name)
get_ipython().system('cp "/content/drive/MyDrive/Colab Notebooks/Grain Demand Supply Prediction.ipynb" "/content/Grain Demand Supply Prediction.ipynb"')


# In[8]:


get_ipython().system('jupyter nbconvert --to python "/content/Grain Demand/Supply Prediction.ipynb"')

