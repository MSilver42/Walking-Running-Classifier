import pandas as pd
import h5py

# walking class is 0
# jumping class is 1

hf = h5py.File('data.h5', 'r')
train = hf.get('data/train')
test = hf.get('data/test')

print(train.shape)
print(test.shape)

for i in range(0, train.shape[0]):
    print(train[i])

