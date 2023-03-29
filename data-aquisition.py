import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)

hdf = pd.HDFStore('data.h5')

temp = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

hdf.put('Mason', temp)
hdf.put('Amy', temp)
hdf.put('Connor', temp)

# display hf structure 
def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))



jumping_df = pd.DataFrame
jumping_files = ['mason-jumping-1.csv', 'Carter_jumping.csv']
for file in jumping_files:
    temp = pd.read_csv(file)
    if jumping_df.empty:
        jumping_df = temp
    else:
        jumping_df = pd.concat([jumping_df, temp])

walking_df = pd.DataFrame
walking_files = ['mason-walking-1.csv', 'Carter_walking.csv']
for file in walking_files:
    temp = pd.read_csv(file)
    if walking_df.empty:
        walking_df = temp
    else:
        walking_df = pd.concat([walking_df, temp])


# add labels
walking_df['class'] = 0
jumping_df['class'] = 1

# combine dataframes
df = pd.concat([walking_df, jumping_df], ignore_index=True)

# replace column names
df.columns = ['Time', 'Accel X', 'Accel Y', 'Accel Z', 'Absolute Accel', 'Class']

hdf.put('data', df, format='table', data_columns=True)

print(hdf.keys())


