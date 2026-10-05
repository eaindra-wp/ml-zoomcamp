from pathlib import Path

import pandas as pd
import numpy as np

def main():
    HERE = Path(__file__).parent.parent
    df = pd.read_csv(HERE / "car_fuel_efficiency_2026.csv")

    print(f'ANSWERS:')
    print(f'1: {pd.__version__}')

    print(f'2: {df['origin'].count()}')

    print(f'3: {df['fuel_type'].unique().__len__()}')

    missing = df.isnull().sum()
    print(f'4: {missing[missing > 0].count()}')

    print(f'5: {df['fuel_efficiency_mpg'].max()}')

    horsepower = df['horsepower']
    hp_med = horsepower.median()
    hp_frequent = horsepower.mode()[0]
    hp_fre_med = df['horsepower'].fillna(hp_frequent).median()
    print(f'6: {hp_med - hp_fre_med}')

    x = (
        df[df['origin'] == 'Asia']
        [['vehicle_weight', 'model_year']]
        .iloc[:7]
        .to_numpy()
    )
    tx = np.transpose(x)
    xtx = np.matmul(tx, x)
    ixtx = np.linalg.inv(xtx)
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    w = ixtx @ tx @ y
    print(f'7: {np.sum(w)}')

if __name__ == "__main__":
    main()
