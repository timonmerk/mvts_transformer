import numpy as np
from matplotlib import pyplot as plt


PATH_DATA = "/Users/Timon/Documents/mvts_transformer/sub_rcs02l.npy"

data = np.load(PATH_DATA)
print(data.shape)
for i in range(data.shape[1]):
    plt.plot(data[:1000, i])
plt.show(block=True)
