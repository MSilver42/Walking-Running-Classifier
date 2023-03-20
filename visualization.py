import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py


# array columns: Time, Accel X, Accel Y, Accel Z, Absolute Accel Class
# array rows: 1 row per 5 seconds of data
# walking class is 0
# jumping class is 1

hf = h5py.File('data.h5', 'r')
train = np.array(hf.get('data/train'))
test = np.array(hf.get('data/test'))

arr = np.concatenate((train, test), axis=0)

figure, axis = plt.subplots(2, 2, figsize=(10, 10))

n=5000

axis[0, 0].scatter(arr[n:n+2000, 0], arr[n:n+2000, 1])
axis[0, 0].set_title('Accel X vs Class')

axis[0, 1].scatter(arr[n:n+2000, 0], arr[n:n+2000, 2])
axis[0, 1].set_title('Accel Y vs Class')

axis[1, 0].scatter(arr[n:n+2000, 0], arr[n:n+2000, 3])
axis[1, 0].set_title('Accel Z vs Class')

axis[1, 1].scatter(arr[n:n+2000, 0], arr[n:n+2000, 4])
axis[1, 1].set_title('Absolute Accel vs Class')

plt.show()





