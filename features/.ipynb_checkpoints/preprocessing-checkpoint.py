import numpy as np
import tqdm
import pandas as pd

def extract_weekday(dt64):    
    # Extract the day of the week (0 is Monday, 6 is Sunday)
    return dt64.astype('datetime64[h]').astype(object).weekday()

def extract_weekdays(dt64_arr):    
    return np.array([extract_weekday(dt64) for dt64 in dt64_arr])
    
def extract_hour(dt64):
            
    # Convert numpy.datetime64 to a datetime object
    dt = dt64.astype('datetime64[m]').astype(object)
    
    # Extract the hour
    hour = dt.hour
    
    # Extract the quarter hour
    quarter_hour = (dt.minute // 15)/4
    
    return hour + quarter_hour

def extract_hours(dt64_arr):    
    return np.array([extract_hour(dt64) for dt64 in dt64_arr])


def extract_days_since_2020(dt64):
    # Reference date as January 1, 2020
    ref_date = np.datetime64('2020-01-01')
    # Calculate days since 2020-01-01
    return (dt64 - ref_date).astype('timedelta64[D]').astype(int)

def extract_days_since_2020_array(dt64_arr):
    # Apply the extract_days_since_2020 function on each element
    return np.array([extract_days_since_2020(dt64) for dt64 in dt64_arr])


def get_person_data(raw_data, p_idx):

    n_seq = len(raw_data[p_idx])

    assert n_seq, f'no data for p_idx {p_idx}'
    
    dfs = []
    
    for i in tqdm.tqdm(range(n_seq)):
    
        tlin, fitbit = raw_data[p_idx][i].values()
    
        seq = np.c_[extract_days_since_2020(tlin), extract_weekdays(tlin), extract_hours(tlin), fitbit]
        
        df = pd.DataFrame(seq, columns=['index','day','hour','act','hr'])
        dfs.append(df)  

    return pd.concat(dfs)

def get_nan_imputed_array(dfs):

    hours = sorted(dfs.hour.unique())
    X_imputed = []
    for i in dfs['index'].unique():
    
        X_imputed.append([])
        
        day_index = dfs['index']==i
    
        for hour in hours:
    
            x_imputed = np.nan        
            if hour in dfs[day_index]['hour'].values:
                x_imputed = dfs[day_index][dfs[day_index]['hour'] == hour]['act']                            
            X_imputed[-1].append(x_imputed)
            
    return np.array(X_imputed).astype(float)


def pixel_to_torch(all_data):

    hours, hours_y, weekdays, days, days_y, x, y = [],[],[],[],[], [], []
    
    for i, sample in enumerate(all_data):

        if i == 0:
            subj_id = int(np.unique(sample[4]))
        else:
            if int(np.unique(sample[4])) != subj_id:
                print(f'finished {subj_id}')
                break
            
        hours.append(extract_hours(sample[2]))
        hours_y.append(extract_hours(sample[2]))
        weekdays.append(extract_weekdays(sample[2]))
        days.append(extract_days_since_2020_array(sample[3]))
        days_y.append(extract_days_since_2020_array(sample[3]))
        x.append(sample[0])
        y.append(sample[1])
            
    return hours, hours_y, weekdays, days, days_y, np.squeeze(np.array(x)), np.squeeze(np.array(y))
