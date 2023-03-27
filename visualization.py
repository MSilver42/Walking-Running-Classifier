import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# array columns: Time, Accel X, Accel Y, Accel Z, Absolute Accel Class
# array rows: 1 row per 5 seconds of data
# walking class is 0
# jumping class is 1

df = pd.read_hdf('data.h5', 'data')
print(df.head())

rows_per_chunk = int(round(5/(df['Time'][1]-df['Time'][0])))


