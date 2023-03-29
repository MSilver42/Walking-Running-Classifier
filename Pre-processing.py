import pandas as pd

df = pd.read_hdf('data.h5', 'data')
rows_per_chunk = int(round(5/(df['Time'][1]-df['Time'][0])))


# split data by class
walking_df = df[df['Class'] == 0]
walking_df = walking_df.reset_index(drop=True)
walking_df = walking_df.drop(['Class'], axis=1)

jumping_df = df[df['Class'] == 1]
jumping_df = jumping_df.reset_index(drop=True)
jumping_df = jumping_df.drop(['Class'], axis=1)

# delete 1st two seconds of data
walking_df = walking_df[walking_df['Time'] > 2]
walking_df = walking_df.reset_index(drop=True)
jumping_df = jumping_df[jumping_df['Time'] > 2]
jumping_df = jumping_df.reset_index(drop=True)

# delete last two seconds of data
walking_df = walking_df[walking_df['Time'] < walking_df['Time'][-1]-2]
walking_df = walking_df.reset_index(drop=True)
jumping_df = jumping_df[jumping_df['Time'] < jumping_df['Time'][-1]-2]
jumping_df = jumping_df.reset_index(drop=True)

# split data into 5 second chunks and flatten each chunk into a single row
walking_df = walking_df.groupby(walking_df.index // rows_per_chunk)

