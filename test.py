import numpy as np


ar = np.array([[1,4], [2,2], [3,2], [4,2], [5,2], [6,2], [7,2], [8,2], [8,2], [10,2], [11,2], [12,2]])

print(ar)

sp = np.split(ar, 4)

print(len(sp),' -- len')
print(sp)
print('last', ar[-2:])