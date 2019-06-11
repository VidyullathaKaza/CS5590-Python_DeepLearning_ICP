import numpy as np
x = np.random.randint(1, 21, size=15, dtype='I')
print("Original array:")
print(x)
x[x.argmax()] = 0
print("Updated Array by replacing maximum number by ZERO:")
print(x)

