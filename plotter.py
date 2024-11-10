# Let's first take a look at the data to understand its structure and identify the columns related to PM2.5 and device IDs.

import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
#file_path = 'C:\\Users\\Greg\\Desktop\\formatted_files\\compare_data\\Marina_Mentou_Experiments - data (8).csv'
#file_path = 'C:\\Users\\Greg\\Desktop\\formatted_files\\compare_data\\Marina_Mentou_Experiments - data (1).csv'
#data = pd.read_csv(file_path, low_memory = False)
file_path = 'C:\\Users\\Greg\\Downloads\\Marina_Mentou_Experiments - data (8).csv'
data = pd.read_csv(file_path, low_memory=False, decimal=',', thousands=',')
#data = pd.read_excel(file_path)

# Convert the 'MC_PM25' to numeric (removing commas if present) and handle non-numeric values
data['MC_PM25'] = pd.to_numeric(data['MC_PM25'], errors='coerce')

# Convert the 'Datetime' column to datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

# Drop rows with missing PM2.5 or Datetime values
data_clean = data.dropna(subset=['MC_PM25', 'Datetime'])

# Extract the date from the 'Datetime' column for easier filtering
data_clean['Date'] = data_clean['Datetime'].dt.date

device_1_id = '0080E1150510BDE6'
device_2_id = '0080E1150533F233'
device_3_id = '0080E1150510B77B'
device_4_id = '0080E1150533ECF3'

device_1_data = data_clean[data_clean['Device_ID'] == device_1_id]
device_2_data = data_clean[data_clean['Device_ID'] == device_2_id]
device_3_data = data_clean[data_clean['Device_ID'] == device_3_id]
device_4_data = data_clean[data_clean['Device_ID'] == device_4_id]

# Merge the two datasets based on the Datetime column to align the timestamps
"""
merged_data = pd.merge(device_1_data[['Datetime', 'MC_PM25']], 
                       device_2_data[['Datetime', 'MC_PM25']], 
                       on='Datetime', 
                       suffixes=('_device_1', '_device_2'))

# Calculate the difference between the PM2.5 values of the two devices
merged_data['PM25_diff'] = merged_data['MC_PM25_device_1'] - merged_data['MC_PM25_device_2']

# Store the differences in an array
pm25_diff_array = merged_data['PM25_diff'].values

# Print the array of differences
print("Array of PM2.5 differences between Device 1 and Device 2:")
print(pm25_diff_array)
"""

# Filter the data for May 10, 2024
specific_date = pd.to_datetime('2024-10-11').date()
daily_data = data_clean[data_clean['Date'] == specific_date]

# Group the data by hour
#daily_data['Hour'] = daily_data['Datetime'].dt.hour
daily_data.loc[:, 'Hour'] = daily_data['Datetime'].dt.hour
mean_pm25_per_device_hour = daily_data.groupby(['Device_ID', 'Hour'])['MC_PM25'].mean()

print(mean_pm25_per_device_hour)
#device_data = daily_data[daily_data['Device_ID'] == '0080E1150510B77B']

#first_valid_index = device_data.index[0]

# Start a new dataset from the first valid row (we assume the data begins here with no earlier information)
#device_data_from_first_valid = device_data.loc[first_valid_index:]

# Set 'Datetime' as the index to use for resampling
#device_data_from_first_valid.set_index('Datetime', inplace=True)
'''
# Create an empty DataFrame to store the modified data
modified_device_data = pd.DataFrame()

# Resample the data into 15-minute chunks
for start_time, group in device_data_from_first_valid.resample('15T'):
    if len(group) > 0:
        # Get the first and last values of PM2.5 in the 15-minute window
        print(len(group))
        first_value = group.iloc[0]['MC_PM25']
        second_value = group.iloc[1]['MC_PM25']
        last_value = group.iloc[-1]['MC_PM25']
        
        # Fill the first minute with the first value, and the remaining minutes with the last value
        group['MC_PM25'] = [first_value] + [second_value] + [last_value] * (len(group) - 2)
        
        # Append the modified group to the new DataFrame
        modified_device_data = pd.concat([modified_device_data, group])

modified_device_data['Datetime'] = pd.to_datetime(modified_device_data['Datetime'])

# Extract the hour from 'Datetime'
modified_device_data['Hour'] = modified_device_data['Datetime'].dt.hour
'''
# Group the resampled data by 'Device_ID' and 'Hour' and calculate the mean PM2.5
#mean_pm25_per_device_hour_resampled = modified_device_data.groupby(['Device_ID', 'Hour'])['MC_PM25'].mean()
# Reset the index to make 'Datetime' a regular column again
#modified_device_data = modified_device_data.reset_index()

#print(mean_pm25_per_device_hour_resampled)

# Plot PM2.5 data per hour for each device on May 10, 2024
plt.figure(figsize=(10, 6))
for device_id, group in daily_data.groupby('Device_ID'):
    plt.plot(group['Datetime'], group['MC_PM25'], label=device_id)
#plt.plot(modified_device_data['Datetime'], modified_device_data['MC_PM25'], label='15_minute 0080E1150510B77B')
# Add labels and legend
plt.title(f'PM2.5 Levels on {specific_date} by Hour and Device ID')
plt.xlabel('Time (Hourly)')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend(title='Device ID')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
