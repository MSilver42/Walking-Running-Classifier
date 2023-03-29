import pandas as pd
import numpy as np

df = pd.read_hdf('data.h5', 'data')
rows_per_chunk = int(round(5/(df['Time'][1]-df['Time'][0])))


# split data by class
walking_df = df[df['Class'] == 0]
walking_df = walking_df.reset_index(drop=True)
walking_df = walking_df.drop(['Class'], axis=1)

jumping_df = df[df['Class'] == 1]
jumping_df = jumping_df.reset_index(drop=True)
jumping_df = jumping_df.drop(['Class'], axis=1)


# remove outliers that are more than 3 standard deviations from the mean
walking_df = walking_df[(np.abs(walking_df['Absolute Accel']-walking_df['Absolute Accel'].mean()) <= (3*walking_df['Absolute Accel'].std()))]
jumping_df = jumping_df[(np.abs(jumping_df['Absolute Accel']-jumping_df['Absolute Accel'].mean()) <= (3*jumping_df['Absolute Accel'].std()))]


# split data into 5 second chunks and flatten each chunk into a single row
flat_walking = None
for i in range(0, len(walking_df), rows_per_chunk):
    if flat_walking is None:
        flat_walking = walking_df.iloc[i:i+rows_per_chunk].values.flatten()
    else:
        try:
            flat_walking = np.vstack([flat_walking, walking_df.iloc[i:i+rows_per_chunk].values.flatten()])
        except:
            pass

flat_jumping = None
for i in range(0, len(jumping_df), rows_per_chunk):
    if flat_jumping is None:
        flat_jumping = jumping_df.iloc[i:i+rows_per_chunk].values.flatten()
    else:
        try:
            flat_jumping = np.vstack([flat_jumping, jumping_df.iloc[i:i+rows_per_chunk].values.flatten()])
        except:
            pass

print(flat_walking.shape)
print(flat_walking[0][:10])

