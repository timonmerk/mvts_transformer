import pandas as pd
import numpy as np
import os

PATH_PKG = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/pkg_data'

subs = np.unique([f[:10] for f in os.listdir("data") if "sub_rcs" in f and ".npy" in f])

def map_sub(sub):

    ts_ = pd.read_csv(f"data/{sub}_time_stamps.csv", header=None, names=["time_stamps"])
    
    #data = np.load(f"data/{sub}.npy")
    df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub[4:]}_pkg.csv"))
    # map df_pkg pkg_dt to ts_ timestamps
    df_pkg["pkg_dt"] = pd.to_datetime(df_pkg["pkg_dt"])
    ts_["time_stamps"] = pd.to_datetime(ts_["time_stamps"])

    # for each timestamp in ts_, find the closest pkg_dt in df_pkg
    # if it's within 2 min of the timestamp, add the row to the data
    # if it's not, add a row with NaNs
    data_pkg = []
    for ts in ts_["time_stamps"]:
        idx = np.argmin(np.abs(df_pkg["pkg_dt"] - ts))
        if abs((df_pkg["pkg_dt"] - ts).iloc[idx].total_seconds()) <= 120:
            data_pkg.append(df_pkg.iloc[idx].values)
        else:
            data_pkg.append([np.nan]*df_pkg.shape[1])

    # create df
    df_pkg_mapped = pd.DataFrame(data_pkg, columns=df_pkg.columns)
    # indices where first column is not NaN
    idx_not_none = df_pkg_mapped.iloc[:, 0].notnull()
    df_pkg_mapped_not_none = df_pkg_mapped.loc[idx_not_none]
    df_pkg_mapped_not_none.to_csv(f"data/{sub}_pkg_mapped.csv", index=True)
    
    #df_pkg_mapped_not_none = pd.read_csv(f"data/{sub}_pkg_mapped.csv", index_col=0)

if __name__ == "__main__":
    #for sub in subs:
    #    map_sub(sub)
    
    from joblib import Parallel, delayed
    Parallel(n_jobs=-1)(delayed(map_sub)(sub) for sub in subs)