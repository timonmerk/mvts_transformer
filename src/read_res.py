import numpy as np
from matplotlib import pyplot as plt


PATH = "/Users/Timon/Documents/mvts_transformer/output/_2024-11-24_15-02-19_93Q/predictions/best_predictions.npz"
predictions = np.load(PATH)

print(predictions.files)
