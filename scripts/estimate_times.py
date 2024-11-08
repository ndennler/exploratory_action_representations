import pandas as pd
import os

search_times = []
choice_times = []

# iterate over folders in ../data/customization_study_data
for folder in os.listdir('../data/customization_study_data'):
    if 'OneDrive' not in folder:
        continue
    # iterate over files in folder
    print(folder)
    # load the data
    searches = pd.read_csv(f'../data/customization_study_data/{folder}/searches.csv')
    choices = pd.read_csv(f'../data/customization_study_data/{folder}/choices.csv')
   
    # Add a 'source' column to indicate the source (searches or choices)
    searches['source'] = 'searches'
    choices['source'] = 'choices'
    
    # Concatenate the two dataframes, keeping only the 'time' and 'source' columns
    combined_df = pd.concat([searches[['time', 'source']], choices[['time', 'source']]])
    combined_df = combined_df.sort_values('time')
    
    # Calculate the time difference between consecutive rows
    combined_df['time_diff'] = combined_df['time'].diff()
    
    # Remove the first row since it will have a NaT value for time_diff
    combined_df = combined_df.dropna(subset=['time_diff'])
    time_spent = combined_df.groupby('source')['time_diff'].sum()

    if 'searches' not in time_spent:
        time_spent['searches'] = 0
    if 'choices' not in time_spent:
        time_spent['choices'] = 0

    search_times.append(time_spent['searches'])
    choice_times.append(time_spent['choices'])
    print(time_spent)

print(f'Average search time: {sum(search_times)/(1*len(search_times))}')
print(f'Average choice time: {sum(choice_times)/(1*len(choice_times))}')