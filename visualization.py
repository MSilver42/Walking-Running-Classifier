import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# array columns: Time, Accel X, Accel Y, Accel Z, Absolute Accel Class
# array rows: 1 row per 5 seconds of data
# walking class is 0
# jumping class is 1

df = pd.read_hdf('data.h5', 'data')
print(df.head())


# plot walking data
walking_df = df[df['Class'] == 0]
walking_df = walking_df.reset_index(drop=True)
walking_df = walking_df.drop(['Class'], axis=1)
axis, fig = plt.subplots(4, sharex=True)
fig[0].plot(walking_df['Time'], walking_df['Accel X'])
fig[0].set_ylabel('Accel X') 
fig[0].set_title('Walking Data')
fig[1].plot(walking_df['Time'], walking_df['Accel Y'])
fig[1].set_ylabel('Accel Y')
fig[2].plot(walking_df['Time'], walking_df['Accel Z'])
fig[2].set_ylabel('Accel Z')
fig[3].plot(walking_df['Time'], walking_df['Absolute Accel'])
fig[3].set_ylabel('Absolute Accel')
fig[3].set_xlabel('Time (s)')


# plot jumping data
jumping_df = df[df['Class'] == 1]
jumping_df = jumping_df.reset_index(drop=True)
jumping_df = jumping_df.drop(['Class'], axis=1)
axis1, fig1 = plt.subplots(4, sharex=True)
fig1[0].plot(jumping_df['Time'], jumping_df['Accel X'])
fig1[0].set_ylabel('Accel X')
fig1[0].set_title('Jumping Data')
fig1[1].plot(jumping_df['Time'], jumping_df['Accel Y'])
fig1[1].set_ylabel('Accel Y')
fig1[2].plot(jumping_df['Time'], jumping_df['Accel Z'])
fig1[2].set_ylabel('Accel Z')
fig1[3].plot(jumping_df['Time'], jumping_df['Absolute Accel'])
fig1[3].set_ylabel('Absolute Accel')
fig1[3].set_xlabel('Time (s)')

plt.show()





