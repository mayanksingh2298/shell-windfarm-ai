from tqdm import tqdm
import numpy as np
import time

# a = np.random.rand(100,540,50)
a = np.random.rand(54000,500)
# b = np.random.rand(100,540,50)
b = np.random.rand(54000,500)

# def mul_3d(a,b):
# 	c = a.reshape(-1)
# 	d = b.reshape(-1)
# 	return (c*d).reshape(a.shape)

# tik = time.time()
# # prd = mul_3d(a,b)
# tok = time.time()

# print("3d broken first time:",tok-tik)

tik = time.time()
prd = a*b
tok = time.time()

print("batch time:",tok-tik)

tik = time.time()
for i in range(a.shape[0]):
    aa = a[i]
    bb = b[i]
    prd = aa*bb
tok = time.time()
print("loop time:",tok-tik)



