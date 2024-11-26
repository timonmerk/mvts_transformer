import numpy as np
import os
import pandas as pd
import tqdm

def check_missing_data(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = None
    return df

sub = "rcs02l"
PATH_PARQUET = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/parquet'

sub_files = np.array(
    [
        (f, int(f[f.find(sub) + len(sub) + 1 : f.find(".parq")]))
        for f in os.listdir(PATH_PARQUET)
        if sub in f
    ]
)
sort_idx = np.argsort(sub_files[:, 1].astype(int))
files_sorted = sub_files[sort_idx, 0]
l_ = []

for f in tqdm.tqdm(files_sorted):#[-20:]:
    df = pd.read_parquet(os.path.join(PATH_PARQUET, f))
    # df = df.astype(object)
    # df.set_index("timestamp", inplace=True)
    df.index = pd.to_datetime(df.index)
    df_r = df.resample("4ms").ffill(limit=1)
    # find indices of 10 second intervals
    # set the start value to rounded full 10 s
    start_ = df_r.index[0].ceil("10s")

    idx_10s = pd.date_range(start=start_, freq="10s", end=df_r.index[-1])
    # iterate through the 10 s intervals and extract the data
    for idx, idx_time in enumerate(idx_10s, start=1):
        if idx == idx_10s.shape[0]:
            break
        t_low = idx_10s[idx - 1]
        t_high = idx_10s[idx]
        df_r_ = df_r.loc[t_low:t_high]

        df_r_f = check_missing_data(df_r_)
        if df_r_f.sum().sum() == 0:
            continue
        df_ = df_r_f[["0-2", "1-3", "8-10", "9-11"]]*10**6
        # check if there is a single columns that is not NaN
        # indexes of NaN values

        if df_.isnull().values.sum() == 0:
            # print(t_high)
            l_.append(df_.iloc[:-1, :]) # leave out next full sample
np.save(f"sub_{sub}.npy", np.array(pd.concat(l_).values, dtype=np.float16))
#yield df_r_f.index[-1], np.array(df_r_f).T