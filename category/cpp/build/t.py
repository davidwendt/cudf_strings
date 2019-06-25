import cucategory
import numpy as np
from numba import cuda
arr = np.array([1,2,3,3,2,1],dtype=np.int32)
#d_arr = cuda.to_device(arr)
cat = cucategory.from_ndarray(arr)
arr = np.array([0,0,0],dtype=np.int32)
cat.keys(arr)
print(arr)
arr = np.array([0,0,0,0,0,0],dtype=np.int32)
cat.values(arr)
print(arr)

