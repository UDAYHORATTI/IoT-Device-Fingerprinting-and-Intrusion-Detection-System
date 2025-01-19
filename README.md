# IoT-Device-Fingerprinting-and-Intrusion-Detection-System
This project aims to fingerprint IoT devices based on their network behavior and device-specific features, then uses this fingerprint to detect potential intrusions or anomalies in a network. By tracking how IoT devices behave in the network, 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time
import random

# Simulate IoT device traffic data (for example, packet sizes, connection durations, etc.)
def generate_device_traffic_data():
    """
    Simulate network traffic data for different IoT devices.
    """
    normal_data = np.random.normal(loc=100, scale=20, size=(100, 3))  # Normal traffic data
    intrusion_data = np.random.normal(loc=500, scale=100, size=(10, 3))  # Intrusion-like activity (anomalous)
    
    data = np.vstack([normal_data, intrusion_data])
    labels = np.array([0] * 100 + [1] * 10)  # 0 for normal, 1 for intrusion
    return pd.DataFrame(data, columns=["packet_size", "connection_duration", "request_rate"]), labels

# Train device fingerprinting model and intrusion detection classifier
def train_intrusion_detection_model(data):
    """
    Train a RandomForestClassifier model for device fingerprinting and intrusion detection.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data_scaled, labels)
    
    return model, scaler

# Monitor real-time device traffic and classify as normal or intrusion
def monitor_device_traffic(model, scaler):
    """
    Monitor IoT device traffic and classify as normal or intrusion in real time.
    """
    while True:
        # Simulate new device traffic data
        new_data = np.random.normal(loc=100, scale=20, size=(1, 3))  # Simulate normal traffic
        
        # Occasionally simulate intrusion-like activity
        if random.random() > 0.95:
            new_data = np.random.normal(loc=500, scale=100, size=(1, 3))  # Intrusion-like activity
        
        # Standardize and classify the new data
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        
        if prediction == 1:
            print(f"[ALERT] Intrusion detected! Traffic: {new_data}")
        else:
            print(f"[INFO] Normal device traffic: {new_data}")
        
        time.sleep(2)  # Simulate real-time monitoring delay

if __name__ == "__main__":
    # Step 1: Generate IoT device traffic data
    print("Generating IoT device traffic data...")
    traffic_data, labels = generate_device_traffic_data()

    # Step 2: Train the device fingerprinting and intrusion detection model
    print("Training device fingerprinting model...")
    model, scaler = train_intrusion_detection_model(traffic_data)

    # Step 3: Monitor device traffic in real-time
    print("Monitoring device traffic for intrusions...")
    monitor_device_traffic(model, scaler)
