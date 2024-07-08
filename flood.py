import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the datasets
flood_history = pd.read_csv('flood history in lagos.csv')
weather_data = pd.read_excel('3.xlsx')

# Ensure 'Date' columns are in datetime format with a specified format
date_format = "%Y-%m-%d"  # Change this format according to your actual date format
flood_history['Date'] = pd.to_datetime(flood_history['Date'], format=date_format)
weather_data['Date'] = pd.to_datetime(weather_data['datetime'], format=date_format)

# Verify the columns of the loaded DataFrames
print("Flood History Columns:", flood_history.columns)
print("Weather Data Columns:", weather_data.columns)

# Ensure the 'Flood' column exists
if 'Flood' not in flood_history.columns:
    raise KeyError("The 'Flood' column is not present in the flood history data.")

# Merge datasets on the 'Date' column
merged_data = pd.merge(weather_data, flood_history[['Date', 'Flood']], on='Date', how='left')

# Fill missing values in the 'Flood' column
#merged_data['Flood'] = merged_data['Flood'].fillna(0)  # Assuming no flood where data is missing

# List all column names in the merged_data DataFrame
print(merged_data.columns)

# Drop unnecessary columns (update this list based on actual columns)
#columns_to_drop = [
#     'name', 'preciptype', 'snow', 'snowdepth', 'windgust',
#     'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
#     'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
#     'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations',
#     'windspeedmax', 'windspeedmin', 'Location', 'Causes', 'Precipitaion',
#     'Precipitation probability', 'Precipitation cover', 'Precipitation Type',
#     'Wind gust', 'Wind speed', 'Wind direction'
# ]

# Drop columns that are present in the DataFrame
# columns_to_drop = [col for col in columns_to_drop if col in merged_data.columns]
# merged_data = merged_data.drop(columns=columns_to_drop)

# Handle missing values by filling with mean for simplicity
#merged_data.fillna(merged_data.mean(), inplace=True)

# Encode categorical variables if any
label_encoder = LabelEncoder()
# Assuming 'Moon phase' is a categorical column, encode it if it exists
if 'Moon phase' in merged_data.columns:
    merged_data['Moon phase'] = label_encoder.fit_transform(merged_data['Moon phase'])

# Normalize/scale the features
scaler = StandardScaler()
X = merged_data.drop(columns=['Date', 'Flood'])
features_scaled = scaler.fit_transform(X)

# Target variable
target = merged_data['Flood']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Model selection
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Example: Predicting on new data (replace with actual new data)
new_data = [[...]]  # Replace with actual new data
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print("Flood Prediction:", prediction)
