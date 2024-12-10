import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
from models.ts_transformer import model_factory
from options import Options
from running import setup
import os
from copy import deepcopy
import pandas as pd
import umap


args = Options().parse()  # `argsparse` object
config = setup(args)  # configuration dictionary

class FeatureExtractor:
    def __init__(self):
        self.extracted_features = None

    def __call__(self, module, input_, output):
        self.extracted_features = output

PATH_BASE = '/Users/Timon/Documents/mvts_transformer/output/Adam_2024-12-06_14-07-27_KFi (1)'
MODEL_PATH = os.path.join(PATH_BASE, 'checkpoints', 'model_best.pth')
PATH_PREDICTIONS = os.path.join(PATH_BASE, 'predictions', 'best_predictions.pickle')
sub = "rcs02r"
with open(PATH_PREDICTIONS, "rb") as f:
    data = pickle.load(f)

# load model
model = model_factory(config, 4, 250)
checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
state_dict = deepcopy(checkpoint['state_dict'])

# check for fine-tuning
# if change_output:
#     for key, val in checkpoint['state_dict'].items():
#         if key.startswith('output_layer'):
#             state_dict.pop(key)
model.load_state_dict(state_dict, strict=False)
model.eval()
PATH_TEST_SUB = f"data/sub_{sub}.npy"
test_data = np.load(PATH_TEST_SUB)

data_labels = pd.read_csv(f"data/sub_{sub}_pkg_mapped.csv", index_col=0)
data_labels["hour"] = pd.to_datetime(data_labels["pkg_dt"]).dt.hour
indices_valid = np.array(data_labels.index)
test_data = test_data[indices_valid, :, :]

num_run = 500
padding_masks = torch.from_numpy(np.ones((num_run, 250)).astype(bool))
res_tr_out = []
for i in np.arange(0, test_data.shape[0]-num_run, num_run):
    print(i)
    X = torch.from_numpy(test_data[i:i+num_run, :, :].astype(np.float32))
    extractor = FeatureExtractor()
    model.transformer_encoder.register_forward_hook(extractor)
    output = model(X, padding_masks)
    model_extracted_features = extractor.extracted_features.detach().numpy()
    model_extracted_features_mean = model_extracted_features.mean(axis=0)
    res_tr_out.append(model_extracted_features_mean)

res_all = np.concatenate(res_tr_out)
np.save("data/res_umap_rcs_02r.npy", res_all)

from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, random_state=42, perplexity=20, n_iter=500, verbose=1)
data_2d = tsne_model.fit_transform(res_all)


#umap_model = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
#data_2d = umap_model.fit(res_all)
data_2d_norm = (data_2d - data_2d.mean(axis=0)) / data_2d.std(axis=0)

data_labels["daytime"] = data_labels["hour"].apply(lambda x: 1 if x > 8 and x < 20 else 0)
plt.figure()
for idx, col_name in enumerate(["pkg_bk", "pkg_dk", "pkg_tremor", "daytime"]):
    plt.subplot(2, 2, idx+1)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], s=1, alpha=0.5, cmap='viridis', c=data_labels[col_name][:data_2d.shape[0]])
    plt.title("UMAP Projection")
    if col_name == "pkg_dk":
        plt.clim(0, 80)
    elif col_name == "pkg_bk":
        plt.clim(5, 50)
    elif col_name == "pkg_tremor":
        plt.clim(0, 1)
    #elif col_name == "hour":
    #    plt.clim(0, 20)
    plt.xlabel("T-SNE 1")
    plt.ylabel("T-SNE 2")
    plt.colorbar()
    plt.title(col_name)
plt.tight_layout()
plt.show(block=True)

plt_ts_idx = 30
for plt_ts_idx in range(50):
    plt.figure()
    plt.subplot(2, 1, 1)
    for i in range(4):
        plt.plot(test_data[plt_ts_idx, :, i] + 4*i, label=f"ch {i}")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [uV]")
    plt.title("Input Time Series")
    plt.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplot(2, 1, 2)
    plt.imshow(model_extracted_features[:, plt_ts_idx, :], aspect="auto")
    plt.xlabel("Time [ms]")
    plt.title("Transformer Encoder Output")
    plt.ylabel("Dimension")
    plt.tight_layout()
    plt.show(block=True)


loss_curve = pd.read_excel(os.path.join(PATH_BASE, 'metrics_Adam.xls'))
plt.figure()
plt.plot(loss_curve['epoch'], loss_curve['loss'], label="loss")
# make log scale
plt.xlim([1, 400])
plt.ylim([0.4, 1])
plt.xlabel("Epoch")
plt.ylabel("MAE Validaiton Loss")
plt.show(block=True)

for i in range(data["targets"][0].shape[0]):
    plt.figure()
    ch_idx = 0
    plt.plot(data["targets"][0][i, :, ch_idx], label="target")
    plt.plot(data["predictions"][0][i, :, ch_idx], label="prediction")
    plt.plot(data["target_masks"][0][i, :, ch_idx], label="mask")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [uV]")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)