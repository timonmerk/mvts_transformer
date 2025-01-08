import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
from models.ts_transformer import model_factory
from matplotlib.backends.backend_pdf import PdfPages
from options import Options
from running import setup
import os
from copy import deepcopy
import pandas as pd
from sklearn import linear_model, metrics, model_selection

subs = np.unique([f[4:10] for f in os.listdir("data/emb_out_pkg") if f.startswith("emb") and "rcs" in f])
d_out = {}
for sub in subs:
    emb_sub = np.load(f"data/emb_out_pkg/emb_{sub}.npy")
    pkg_sub = pd.read_csv(f"data/emb_out_pkg/pkg_{sub}.csv", index_col=0)
    pkg_sub["pkg_dt"] = pd.to_datetime(pkg_sub["pkg_dt"])
    d_out[sub] = {"emb": emb_sub, "pkg": pkg_sub}

# test within patient
X_ = emb_sub.reshape(emb_sub.shape[0], -1)
X_ = emb_sub.mean(axis=1)
y_not_na = ~pkg_sub["pkg_dk"].isna()
X_ = X_[y_not_na]
pkg_sub_ = pkg_sub[y_not_na]

y_pr = model_selection.cross_val_predict(linear_model.LinearRegression(), X_, pkg_sub_["pkg_dk"].values,
                                         cv=model_selection.KFold(n_splits=3, shuffle=True))
y_te = pkg_sub_["pkg_dk"].values
plt.figure()
plt.plot(y_te, label="true")
plt.plot(y_pr, label="pred")
plt.legend()
corr_ = np.corrcoef(y_te, y_pr)[0, 1]
plt.title(f"corr: {corr_}")
plt.show(block=True)


per_ = []
for sub_test in subs:
    X_train = []
    y_train = []
    for sub_train in subs:
        if sub_test == sub_train:
            X_test = d_out[sub_test]["emb"]
            y_test = d_out[sub_test]["pkg"]["pkg_label"].values
            continue
        X_train.append(d_out[sub_train]["emb"])
        y_train.append(d_out[sub_train]["pkg"]["pkg_label"].values)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    clf = linear_model.LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ba = metrics.balanced_accuracy_score(y_test, y_pred)
    per_.append(ba)