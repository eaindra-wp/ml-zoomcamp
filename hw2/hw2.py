import math
from pathlib import Path

import pandas as pd
import numpy as np

def train_linear_regression(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    r_arr = r * np.eye(XTX.shape[0])
    XTX = XTX + r_arr
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full

def calc_linear_regression(X, w_full):
    w_0, w_rest = w_full[0], w_full[1:]
    return w_0 + X.dot(w_rest)

def calc_rmse(y_val, y_pred):
    se = (y_val - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

def prep_dataset(df: pd.DataFrame, 
                 seed: int, 
                 fill_missing_with_mean: bool = False):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].copy()
    missing_col = df.columns[df.isnull().sum() > 0][0]
    mean = df_train[missing_col].mean()
    buffer = mean if fill_missing_with_mean else 0
    df_train = df_train.fillna(buffer)

    df_val = df.iloc[idx[n_train:n_train + n_val]].copy()
    df_val = df_val.fillna(buffer)
    
    df_test = df.iloc[idx[n_train + n_val:]].copy()
    df_test = df_test.fillna(buffer)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    target = 'fuel_efficiency_mpg'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    del df_train[target]
    del df_val[target]
    del df_test[target]

    return df_train, df_val, df_test, y_train, y_val, y_test

def question_3(df: pd.DataFrame, 
               fill_missing_with_mean: bool = False):

    df_train, df_val, df_test, y_train, y_val, y_test = prep_dataset(df, 42, fill_missing_with_mean)
    w_full = train_linear_regression(df_train, y_train)
    y_pred = calc_linear_regression(df_val, w_full)
    return round(calc_rmse(y_val = y_val, y_pred=y_pred), 3)

def question_4(df: pd.DataFrame):
    df_train, df_val, df_test, y_train, y_val, y_test = prep_dataset(df, 42) # default false means zero filling
    r_arr = np.array([0, 0.01, 0.1, 1, 5, 10, 100])

    best_r, best_rmse = -1,  math.inf

    for r in r_arr:
        w_full = train_linear_regression(df_train, y_train, r)
        y_pred = calc_linear_regression(df_val, w_full)
        rmse = round(calc_rmse(y_val = y_val, y_pred=y_pred), 4)
        if rmse < best_rmse:
            best_r, best_rmse = r, rmse

    return best_r, best_rmse


def question_5(df: pd.DataFrame):
    seeds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    rmse_arr = []

    for s in seeds:
        df_train, df_val, df_test, y_train, y_val, y_test = prep_dataset(df, s) # default false means zero filling
        w_full = train_linear_regression(df_train, y_train)
        y_pred = calc_linear_regression(df_val, w_full)
        rmse_arr.append(calc_rmse(y_val = y_val, y_pred=y_pred))

    rmse_arr = np.array(rmse_arr)
    return round(np.std(rmse_arr), 3)

def question_6(df: pd.DataFrame):
    df_train, df_val, df_test, y_train, y_val, y_test = prep_dataset(df, 9) # default false means zero filling

    df_train = pd.concat([df_train, df_val], ignore_index=True)
    y_train = np.concatenate([y_train, y_val])

    w_full = train_linear_regression(df_train, y_train, r=0.001)
    y_pred = calc_linear_regression(df_test, w_full)
    return calc_rmse(y_val = y_test, y_pred=y_pred)


def main():
    HERE = Path(__file__).parent.parent
    df = pd.read_csv(HERE / "car_fuel_efficiency_2026.csv")

    eda = ['engine_displacement',
            'horsepower',
            'vehicle_weight',
            'model_year',
            'fuel_efficiency_mpg']

    df_cpy = df[eda]
    missing_col_name = df_cpy.columns[df_cpy.isnull().sum() > 0][0]
    print(f'1. {missing_col_name}')

    print(f'2. {(df_cpy[missing_col_name].median())}')

    rmse_zero = question_3(df_cpy)
    rmse_mean = question_3(df_cpy, fill_missing_with_mean=True)
    print(f'3. rmse_with_zero_filling = {rmse_zero}, rmse_with_mean_filling = {rmse_mean}')

    print(f'4. best_r = {question_4(df_cpy)[0]}')

    print(f'5. std = {question_5(df_cpy)}')

    print(f'6. {question_6(df_cpy):.4g}')

if __name__ == "__main__":
    main()