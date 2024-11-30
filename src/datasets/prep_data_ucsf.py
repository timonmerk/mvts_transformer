import numpy as np
import os
import pandas as pd
import tqdm

def check_missing_data(df):
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = None
    return df

PATH_PARQUET = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/parquet'
subs = np.unique([f[:6] for f in os.listdir(PATH_PARQUET)])

def run_sub(sub):
    sub_files = np.array(
        [
            (f, int(f[f.find(sub) + len(sub) + 1 : f.find(".parq")]))
            for f in os.listdir(PATH_PARQUET)
            if sub in f
        ]
    )
    sort_idx = np.argsort(sub_files[:, 1].astype(int))
    files_sorted = sub_files[sort_idx, 0]
    cnt_ID = 0
    arr_concat = np.array([])
    try:
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
                #df_ = df_r_f[["0-2", "1-3", "8-10", "9-11"]]*10**6
                # sum elements in each column that is not None or NaN
                column_sums = df_r_f.apply(lambda x: x.dropna().abs().sum())
                # select the four columns with the highest sums
                top_columns = column_sums.nlargest(4).index
                if (column_sums[top_columns] == 0).any():
                    continue
                df_ = df_r_f[top_columns]

                # check if there is a single columns that is not NaN
                # indexes of NaN values

                if df_.isnull().values.sum() == 0:
                    # print(t_high)
                    df_["ID"] = cnt_ID
                    cnt_ID += 1
                    if cnt_ID > 65000:
                        # stop both for loops and save
                        # throw exception
                        raise Exception("More than 65000 samples")
                    arr_ = np.array(df_.values)

                    arr_ = (arr_ - arr_.mean(axis=0)) / arr_.std(axis=0)
                    arr_ = arr_.astype(np.float16)[:-1,:]
                    arr_[:, 4] = np.float16(cnt_ID)
                    if arr_concat.size == 0:
                        arr_concat = arr_
                    else:
                        arr_concat = np.concatenate((arr_concat, arr_))
                    #l_.append(df_.iloc[:-1, :]) # leave out next full sample
    except Exception as e:
        print(e)
    finally:
        np.save(f"sub_{sub}.npy", arr_concat)
    #yield df_r_f.index[-1], np.array(df_r_f).T

if __name__ == "__main__":
    #run_sub(subs[0])
    ESTIMATE_NPY = False
    if ESTIMATE_NPY:
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, verbose=10)(delayed(run_sub)(sub) for sub in subs)
    
    # read all npy files and concatenate them
    arr_concat = np.array([])
    for sub in subs:
        arr_ = np.load(f"sub_{sub}.npy")
        if arr_concat.size == 0:
            arr_concat = arr_
        else:
            arr_concat = np.concatenate((arr_concat, arr_))
    # create an array with incrementing steps of 0.01, and repeat each value 250 times
    #NUM_TIMESERIES = arr_concat.shape[0] / 250
    #np.arange(0, NUM_TIMESERIES / 100, 0.01)
    #arr_concat[:, 4] = np.repeat(np.arange(0, 65, 0.1), 1000)
    
    np.save("all_subs.npy", arr_concat[:, :4])