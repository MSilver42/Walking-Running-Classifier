import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split

hf = h5py.File('data.h5', 'w')

g1 = hf.create_group('data')
g4 = hf.create_group('Mason')
g5 = hf.create_group('Amy')
g6 = hf.create_group('Connor')



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
jumping_files = ['mason-jumping-1.csv']
for file in jumping_files:
    temp = pd.read_csv(file)
    if jumping_df.empty:
        jumping_df = temp
    else:
        jumping_df = jumping_df.append(temp)

walking_df = pd.DataFrame
walking_files = ['mason-walking-1.csv']
for file in walking_files:
    temp = pd.read_csv(file)
    if walking_df.empty:
        walking_df = temp
    else:
        walking_df = walking_df.append(temp)

rows_per_chunk = int(round(5/(walking_df['Time (s)'][1]-walking_df['Time (s)'][0])))

# take average of each chunk
# walking_df = walking_df.groupby(np.arange(len(walking_df))//rows_per_chunk)
# jumping_df = jumping_df.groupby(np.arange(len(jumping_df))//rows_per_chunk)

# add labels
walking_df['class'] = 0
jumping_df['class'] = 1


print("Walking Dataframe")
print(walking_df.head())
print(walking_df.shape)
print("Jumping Dataframe")
print(jumping_df.head())
print(jumping_df.shape)

# combine dataframes
df = walking_df.append(jumping_df)

# split data into train and test
train, test = train_test_split(df, test_size=0.2, random_state=42)




# add train data to h5 file
hf.create_dataset('data/train', data=train)
# add test data to h5 file
hf.create_dataset('data/test', data=test)

with h5py.File('data.h5', 'r') as i:
    print(i)
    h5_tree(i)

hf.close()
