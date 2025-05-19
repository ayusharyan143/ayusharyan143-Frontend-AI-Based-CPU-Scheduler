import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from stable_baselines3 import PPO

# Load and prepare data
df = pd.read_csv(r"C:\Users\ayush\Desktop\OS_PBL_Project\AI-Based-CPU-Scheduler-main\approach3\burst_time_prediction\process_data.csv")

# Define feature names explicitly
FEATURE_NAMES = [
    'io_write_bytes',
    'num_ctx_switches_voluntary',
    'cpu_percent',
    'io_read_bytes',
    'io_read_count',
    'io_write_count'
]

# Create feature matrix
X = df[FEATURE_NAMES]

# Create target variable (burst time)
y = df['cpu_times_user'] + df['cpu_times_system']

# Handle missing values
X = X.fillna(0)

# Feature scaling with proper feature names
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=FEATURE_NAMES
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Results:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': FEATURE_NAMES,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save the model and scaler
joblib.dump(rf_model, 'burst_time_predictor.joblib')
joblib.dump(scaler, 'feature_scaler.joblib')

print("\nModel and scaler saved successfully!")

def predict_burst_time(process_features):
    loaded_model = joblib.load('burst_time_predictor.joblib')
    loaded_scaler = joblib.load('feature_scaler.joblib')
    
    # Create DataFrame with correct feature names
    features_df = pd.DataFrame([process_features], columns=FEATURE_NAMES)
    
    # Scale features
    scaled_features = pd.DataFrame(
        loaded_scaler.transform(features_df),
        columns=FEATURE_NAMES
    )
    
    return loaded_model.predict(scaled_features)[0]

# Example usage with proper feature order
example_process = [
    25000,   # io_write_bytes
    100,     # num_ctx_switches_voluntary
    50.0,    # cpu_percent
    50000,   # io_read_bytes
    1000,    # io_read_count
    500      # io_write_count
]

predicted_burst_time = predict_burst_time(example_process)
print(f"\nExample Prediction:")
print(f"Predicted burst time: {predicted_burst_time:.2f} seconds")

try:
    print("Attempting to load models...")
    model_path = "rl_scheduler_model.zip"  # Use the zip file instead of the directory
    print(f"Loading RL model from: {model_path}")
    
    rl_model = PPO.load(model_path)
    print("RL model loaded successfully")
    
    print("Loading burst predictor...")
    burst_predictor = joblib.load('burst_time_predictor.joblib')
    print("Burst predictor loaded successfully")
    
    print("Loading feature scaler...")
    feature_scaler = joblib.load('feature_scaler.joblib')
    print("Feature scaler loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print("Full traceback:")
    print(traceback.format_exc())

