import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
from torch import nn
from models.ts_transformer import model_factory
from matplotlib.backends.backend_pdf import PdfPages
from options import Options
from running import setup
import os
from copy import deepcopy
import pandas as pd

args = Options().parse()  # `argsparse` object
config = setup(args)  # configuration dictionary
config["d_model"] = 128
config["dim_feedforward"] = 256
config["num_layers"] = 3

class FeatureExtractor:
    def __init__(self):
        self.extracted_features = None

    def __call__(self, module, input_, output):
        self.extracted_features = output

#PATH_BASE = "/Users/Timon/Documents/mvts_transformer/output"
PATH_BASE = "output"
model_name = "0.3m_Adam" 
device = torch.device("cuda")
#device = torch.device("mps")

#model = nn.DataParallel()
model = model_factory(config, 4, 250)
model = torch.nn.DataParallel(model)
MODEL_PATH = os.path.join(PATH_BASE, model_name, 'checkpoints', 'model_best.pth')
checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
state_dict = deepcopy(checkpoint['state_dict'])
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('model.', '')  # Remove 'module.' prefix
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
# get mps device

model = model.to(device)
model.eval()
extractor = FeatureExtractor()
if type(model) == torch.nn.DataParallel:
    model.module.transformer_encoder.register_forward_hook(extractor)
else:
    model.transformer_encoder.register_forward_hook(extractor)

subs = np.unique([f[4:10] for f in os.listdir("data") if f.startswith("sub") and "rcs" in f])
for sub in subs:
    print(sub)
    PATH_TEST_SUB = f"data/sub_{sub}.npy"
    test_data = np.load(PATH_TEST_SUB)

    data_labels = pd.read_csv(f"data/sub_{sub}_pkg_mapped.csv", index_col=0)
    data_labels["pkg_dt"] = pd.to_datetime(data_labels["pkg_dt"])
    pkg_unique = data_labels["pkg_dt"].unique()

    # get predictions
    emb_l = []
    pkg_l = []
    for idx, unique_dt in enumerate(pkg_unique):
        idxs = data_labels[data_labels["pkg_dt"] == unique_dt].index
        padding_masks = torch.from_numpy(np.ones((idxs.shape[0], 250)).astype(bool))
        padding_masks = padding_masks.to(device)
        X = torch.from_numpy(test_data[idxs, :, :].astype(np.float32))
        X = X.to(device)
        output = model(X, padding_masks)
        model_extracted_features = extractor.extracted_features.cpu().detach().numpy()
        emb_l.append(model_extracted_features.mean(axis=1))
        pkg_l.append(data_labels.loc[idxs[0]])
        if idx % 100 == 0:
            print(idx)
    emb = np.array(emb_l)
    pkg = pd.DataFrame(pkg_l)
    PATH_SAVE = f"data/emb_out_pkg/emb_{sub}.npy"
    np.save(PATH_SAVE, emb)
    pkg.to_csv(f"data/emb_out_pkg/pkg_{sub}.csv")