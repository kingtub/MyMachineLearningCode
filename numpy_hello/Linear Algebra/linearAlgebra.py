import numpy as np
from numpy import linalg

b = np.random.rand(4,4)
print(b)
invb = linalg.pinv(b)
print(invb)
print(b.dot(invb))