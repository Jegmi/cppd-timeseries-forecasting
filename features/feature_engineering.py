def get_features_and_labels(vi, bin_size = 60):
        
    x = vi[0].reshape(-1,bin_size).astype(int)
    y = vi[1].reshape(-1,bin_size).astype(int)
    x_time = extract_hours(vi[2]).reshape(-1,bin_size)
    y_time = extract_hours(vi[3]).reshape(-1,bin_size)
    x_weekday = extract_weekdays(vi[2]).reshape(-1,bin_size)
    y_weekday = extract_weekdays(vi[3]).reshape(-1,bin_size)
    x_id = vi[4].reshape(-1,bin_size).astype(int)
    y_id = vi[5].reshape(-1,bin_size).astype(int)
    #x_absolute_day = extract_days_since_2020_array(vi[6])
    assert len(set(list(x_id.reshape(-1)))) == 1, 'multiple subj_ids per segment: error!'
        
    feat = [
        np.sum(x == 1, axis=1),
        np.sum(x == 2, axis=1),
        np.sum(x > 2, axis=1),
        #np.sum(x == 4, axis=1),
        np.sum(np.abs(np.diff(x,1)) > 0, 1),        
        np.array([x_time_i[0] for x_time_i in x_time]), # collaspse to first?
        np.array([x_weekday_i[0] for x_weekday_i in x_weekday]),
        x_id[:,0],
    ]
    
    label = np.array([np.sum(y, axis=1) == bin_size])
    
    return np.array(feat),np.squeeze(label)

def to_sklearn(val, bin_size, stride):
    X,Y=[],[]
    for i in tqdm.tqdm(range(len(val))):
        if i % stride == 0:
            x,y = get_features_and_labels(val[i],bin_size)
            X.append(x)
            Y.append(y)
    return X,Y