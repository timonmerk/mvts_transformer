import numpy as np
from matplotlib import pyplot as plt
import pickle



PATH = "/Users/Timon/Documents/mvts_transformer/output/_2024-11-27_10-45-53_qQf/predictions/best_predictions.pickle"

with open(PATH, "rb") as f:
    data = pickle.load(f)

for i in range(data["targets"][0].shape[0]):
    plt.figure()
    ch_idx = 0
    plt.plot(data["targets"][0][i, :, ch_idx], label="target")
    plt.plot(data["predictions"][0][i, :, ch_idx], label="prediction")
    plt.plot(data["target_masks"][0][i, :, ch_idx], label="mask")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)