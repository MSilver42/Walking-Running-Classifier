import numpy as np
import matplotlib.pyplot as plt
import h5py

hf = h5py.File('data.h5', 'w')

g1 = hf.create_group('data')
g2 = hf.create_group('data/train')
g3 = hf.create_group('data/test')

add_new = input("Add new user? (y/n): ")
if add_new == "y":
    # enter name of group member
    name = input("Enter name: ")
    g4 = hf.create_group(name)

add_new = input("Add new data? (y/n): ")
if add_new == "y":
    file_name = input("Enter file name: ")
    data = np.loadtxt(file_name)
    data_class = input("Enter run or walk: ")
    if data_class == "run":
        # split data into 5 second chunks
        data = np.split(data, 5)



""" # display hf structure 
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

with h5py.File('data.h5', 'r') as i:
    print(i)
    h5_tree(i)
"""