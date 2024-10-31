"""
prediction_horizon = 2 # min
look_back_window = 10

use_time_features=True
all_data = Dataset_CPP('./', features='activity',
                         data_path='private/cpp/20240820pkls/min15.pkl', 
                         size=[look_back_window, 0, prediction_horizon], scale=False, 
                   use_time_features=use_time_features,split='all')

"""                   

import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
#from src.data.timefeatures import time_features
import warnings


def _torch(*args, **kwargs): # just in case I want to go back to a torch dataset
    return list(args)


warnings.filterwarnings('ignore')
class Dataset_CPP():
    def __init__(self, root_path, split='train', size=[384//8, 96//8, 96//8],
                 features=None, data_path='private/cpp/20240820pkls/min5.pkl',
                 target='heart_rate', scale=True, timeenc=0, use_time_features=False, 
                 train_split=0.7, test_split=0.2, do_read_data=True,
                 ):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # init
        assert split in ['train', 'test', 'val', 'all']
        self.split = split

        feat_name_to_col_idx = {'activity':[0],'ac':[0], 'heart':[1], 'heart_rate':[1], 'hr':[1]}
        self.features = feat_name_to_col_idx[features] if features else None 
        self.scale = scale
        self.timeenc = timeenc
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split # wont need numbers

        self.root_path = root_path
        self.data_path = data_path
        if do_read_data:                
            self.__read_data__()


    def _load_file(self):
        filepath = os.path.join(self.root_path, self.data_path)
        
        with open(filepath, 'rb') as f:
            data_raw = pickle.load(f)

        return data_raw
        
    
    def __read_data__(self):
        self.scaler = StandardScaler()
                
        data_raw = self._load_file()

        '''
        df_raw = dict(subject : [ dict('datetime':, 'values':) , dict(), .. ])
        '''
        
        subjects = list(data_raw.keys())
        num_subj = len(subjects)
        
        train_ids = []
        val_ids = []
        test_ids = []        
        for i, subj in enumerate(subjects):
            if i < int(num_subj * self.train_split):
                train_ids.append(subj)
            elif i < int(num_subj * (self.train_split + self.test_split)):
                test_ids.append(subj)
            else:
                val_ids.append(subj)
        print('num of subjects in train, val, test',len(train_ids), len(val_ids), len(test_ids))
        
        if self.split == 'train':
            ids = train_ids
        elif self.split == 'test':
            ids = test_ids
        elif self.split == 'val':
            ids = val_ids
        elif self.split == 'all':
            ids = subjects
        
        # read out data
        
        # require values[t:t+margin] to exist!
        margin = self.pred_len + self.seq_len
    
        # lists of sequences (flatten over subjects)
        datetime, data, subj_ids = self._make_list_of_sequences_from_ids(data_raw, ids, margin)
        
        # debugging
        self.datetime_list = datetime.copy()
        
        # across all sequeces and subjects, get map: dset-index -> timestep t within margin
        self.idx_to_time = self._map_idx_to_time(data, margin)
        self.idx_max = max(list(self.idx_to_time.keys()))

        # concat lists
        data = np.concatenate(data)
        datetime = np.concatenate(datetime)
        subj_ids = np.concatenate(subj_ids)
        
        if self.scale:            
            if self.split != 'train':
                data_train = np.concatenate(self._make_list_of_sequences_from_ids(data_raw, train_ids, margin)[1])                            
                self.scaler.fit(data_train)                
                data = self.scaler.transform(data)
            else:
                data = self.scaler.fit_transform(data)

        self.data_x = data
        self.data_y = data
        self.data_ids = subj_ids
        self.data_stamp = datetime
                        
        print(self.total_data_lost, 'timesteps lost as fraction',self.total_data_lost/(self.idx_max+self.total_data_lost))                    
        

        
    def __getitem__(self, index):
        # index has to jump over seq!
        
        s_begin = self.idx_to_time[index] # includes jumps to ensure margins
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x_id = self.data_ids[s_begin:s_end]
        seq_y_id = self.data_ids[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_id, seq_y_id)
        else: return _torch(seq_x, seq_y)

    def __len__(self):        
        return self.idx_max

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def _make_list_of_sequences_from_ids(self, data_raw, ids, margin):
        datetime = []
        values = []  
        subj_ids = []
        total_data_lost = 0
        for subj_id in ids:
            num_seq = len(data_raw[subj_id])
            for i in range(num_seq):

                datetime_i = data_raw[subj_id][i]['datetime']
                if len(datetime_i) > margin:                    
                    datetime.append(datetime_i)
                    subj_ids.append([subj_id]*len(datetime_i))                   
                    val_i_selected = self.select_features(data_raw[subj_id][i]['values'])
                    values.append(val_i_selected)
                else:
                    #print('skip (subj, seq)', subj_id, i, 'with T:',len(datetime_i))
                    total_data_lost += len(datetime_i)
              
        self.total_data_lost = total_data_lost
        return datetime, values, subj_ids

    def _map_idx_to_time(self, values, margin):        
        dset_idx = 0
        t_seq_start = 0
        idx_to_time = []
        # loop over sequences
        for val in values: 
            T = len(val)            
            for t in range(T): 
                t_margin = T - margin                
                assert t_margin > 0, f't_margin: {t_margin}, but must be >0'

                if t < t_margin: # include timesteps within margin
                    idx_to_time.append([dset_idx, t_seq_start + t])    
                    dset_idx += 1
                else: # set time-stamp to start of next sequence
                    t_seq_start += T
                    break
        return dict(idx_to_time)
    
    
    def select_features(self, values):       
        # select activity or heart, or keep both
        if self.features is not None: return values[:,self.features]            
        else: return values

