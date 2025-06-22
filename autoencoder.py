import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Generate normal data (sine wave)
t = np.linspace(0, 100, 1000)
normal_data = np.sin(t) + 0.1 * np.random.randn(1000)

# Initial data analysis
df = pd.DataFrame({'Time': t, 'Normal Data': normal_data})

# Introduce anomalies (random spikes)
anomaly_indices = np.random.choice(1000, 50, replace=False)
anomalies = normal_data.copy()
anomalies[anomaly_indices] += np.random.uniform(3, 5, size=50)

# Combine normal and anomaly data
data = np.concatenate([normal_data, anomalies]).reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split data into training (normal only) and testing (normal + anomalies)
X_train, X_test = train_test_split(data_scaled[:1000], test_size=0.2, random_state=42)
X_test = np.concatenate([X_test, data_scaled[1000:]])  # Add anomalies to test set

# Define Autoencoder architecture
input_dim = X_train.shape[1]
autoencoder = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

# Compile model
autoencoder.compile(optimizer='adam', loss='mse')

# Train model on normal data
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, validation_data=(X_test, X_test))

# Compute reconstruction loss (MSE)
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.abs(X_test - X_test_pred), axis=1)

# Set threshold (mean + 3 * std)
threshold = np.mean(mse) + 3 * np.std(mse)

# Identify anomalies
anomalies = mse > threshold

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Loss")
plt.axhline(y=threshold, color='r', linestyle='--', label="Anomaly Threshold")
plt.legend()
plt.title("Anomaly Detection using Autoencoder")
plt.show()

# Print number of detected anomalies
print(f"Total anomalies detected: {np.sum(anomalies)}")