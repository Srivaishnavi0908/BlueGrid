import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

num_rows = 10  # total number of data points

iot_data = np.random.randint(5000, 10000, size=5)
iot_data = np.pad(iot_data, (0, num_rows - len(iot_data)), 'constant')

satellite_data = np.random.randint(20000, 50000, size=3)
satellite_data = np.pad(satellite_data, (0, num_rows - len(satellite_data)), 'constant')

weather_data = np.random.randint(0, 50, size=num_rows)

citizen_reports = np.random.randint(0, 6, size=num_rows)

data = pd.DataFrame({
    'IoT': iot_data,
    'Satellite': satellite_data,
    'Weather': weather_data,
    'Citizen': citizen_reports
})

print("Sample Hybrid Data:\n", data)
data['Available_Water'] = data['IoT'] + data['Satellite'] + data['Weather']*100
data['Shortage'] = (data['Available_Water'] < 50000).astype(int)
X = data[['IoT', 'Satellite', 'Weather', 'Citizen']]
y = data['Shortage']
model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
data['Predicted_Shortage'] = [1 if p>0.5 else 0 for p in predictions]
for idx, row in data.iterrows():
    if row['Predicted_Shortage'] == 1:
        print(f"ALERT: Predicted water shortage at data point {idx}!")

print("\n--- Prototype Summary ---")
print(f"Total data points: {len(data)}")
print(f"Shortages detected: {data['Predicted_Shortage'].sum()}")
print(f"Accuracy of simple model (simulated): {round((len(data) - abs(data['Shortage'] - data['Predicted_Shortage']).sum()) / len(data) * 100,2)}%")
